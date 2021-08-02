import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
# Simple Stock app Testing from tutorial

shown are the stork closing price and volume of Google!
""")

ticker_symbol = 'AAPL'

ticker_data = yf.Ticker(ticker_symbol)
ticker_df = ticker_data.history(
    period='id', start='2010-5-31', end='2020-5-31')

st.write('### Closing Price')
st.line_chart(ticker_df.Close)

st.write('### Volume Price')
st.line_chart(ticker_df.Volume)
