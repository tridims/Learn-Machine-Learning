import os, sys, time, random
import numpy as np
from IPython.display import clear_output
from contextlib import closing
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete
from time import sleep

MAP = [
    "+-------------------+",
    "|R| | : | : | : : :G|",
    "| | | | : : | : : : |",
    "| | : | : : | : | : |",
    "| | : | : : | : | : |",
    "| | : | : : | : | : |",
    "| | : | : |B| : | : |",
    "| : : | : : : : | : |",
    "| | : | : : : : | : |",
    "| | : | : : : : | : |",
    "|Y| : : : : | : : : |",
    "+-------------------+",
]


class DeliveryRobotEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.locs = locs = [(0, 0), (0, 9), (9, 0), (5, 5)]

        num_states = 2000  # 10 * 10 * 5 * 4
        num_rows = 10
        num_columns = 10
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        P = {state: {action: [] for action in range(num_actions)} for state in range(num_states)}

        for row in range(num_rows):
            for col in range(num_columns):
                for goods_idx in range(len(locs) + 1):
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, goods_idx, dest_idx)
                        if goods_idx < 4 and goods_idx != dest_idx:
                            initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            new_row, new_col, new_goods_idx = row, col, goods_idx
                            reward = -1
                            done = False
                            robot_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if goods_idx < 4 and robot_loc == locs[goods_idx]:
                                    new_goods_idx = 4
                                else:  # goods not at location
                                    reward = -10
                            elif action == 5:  # dropoff
                                if (robot_loc == locs[dest_idx]) and goods_idx == 4:
                                    new_goods_idx = dest_idx
                                    done = True
                                    reward = 30
                                elif (robot_loc in locs) and goods_idx == 4:
                                    new_goods_idx = locs.index(robot_loc)
                                else:  # dropoff at wrong location
                                    reward = -10
                            new_state = self.encode(new_row, new_col, new_goods_idx, dest_idx)
                            P[state][action].append((1.0, new_state, reward, done))

        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, robot_row, robot_col, goods_loc, dest_idx):
        """
        Berfungsi untuk menencode keadaan lingkungan yang berupa (robot_row, robot_col, lokasi barang, tujuan)
        ke dalam suatu angka diskrit untuk merepresentasikan keadaan lingkungan

        4 data keadaan lingkungan tersebut akan di encode ke dalam nilai diskrit di antara 0 sampai 2000

        """
        i = robot_row
        i *= 10
        i += robot_col
        i *= 5
        i += goods_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i):
        """
        Untuk men-decode angka representasi state kembali ke bentuk (robot_row, robot_col, lokasi_barang, tujuan)
        """
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 10)
        i = i // 10
        out.append(i)
        assert 0 <= i < 10
        return reversed(out)

    def step(self, a):
        assert 0 <= a < 2000
        return super().step(a)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        robot_row, robot_col, goods_idx, dest_idx = self.decode(self.s)

        # Coloring barrier ( '|', '+', '-' )
        for a in out:
            for b in range(len(a)):
                if a[b] == '|' or a[b] == '+' or a[b] == '-':
                    a[b] = utils.colorize(a[b], 'cyan')

        def ul(x):
            return "_" if x == " " else x

        if goods_idx < 4:
            out[1 + robot_row][2 * robot_col + 1] = utils.colorize(
                out[1 + robot_row][2 * robot_col + 1], 'yellow', highlight=True)
            pi, pj = self.locs[goods_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + robot_row][2 * robot_col + 1] = utils.colorize(
                ul(out[1 + robot_row][2 * robot_col + 1]), 'green', highlight=True)

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Selatan", "Utara", "Timur", "Barat", "Ambil Barang", "Taruh Barang"][self.lastaction])
            )

        else:
            outfile.write("\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

env = DeliveryRobotEnv()

q_table = np.load("ModelAgen.npy")

def print_frames(frames):
    """Untuk memprint setiap frame dari gameplay, input berupa list berisi dictionary"""
    for i, frame in enumerate(frames):
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.4)
        if i != len(frames)-1:
            os.system('cls' if os.name == 'nt' else 'clear')

def test_agent():
    epochs = 0
    penalties = 0

    frames = []  # for animation

    done = False
    state = env.reset()

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        }
        )

        epochs += 1

    sleep(3)
    os.system('cls' if os.name == 'nt' else 'clear')
    print_frames(frames)

    print(f"\n\nFinal Result\\\n\tBanyak Langkah yang diambil: {epochs}")
    print(f"\tPenalti Terjadi: {penalties}")



if __name__ == '__main__':
    a = int(input("Berapa kali mau di run ? "))
    for _ in range(a):
        test_agent()