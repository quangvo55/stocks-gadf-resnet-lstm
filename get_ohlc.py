import yfinance as yf
import pandas as pd

# Define the stock ticker and the period for which you want to download data
ticker = 'LUNR'  # Example: Apple Inc.
period = '1y'    # Example: 1 year

# Download the stock data
stock_data = yf.download(ticker, period=period)

# Select only the OHLC columns
ohlc_data = stock_data[['Open', 'High', 'Low', 'Close']]

# Save the OHLC data to a CSV file
ohlc_data.to_csv('ohlc_data.csv')

print("OHLC data has been saved to ohlc_data.csv")