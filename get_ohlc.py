import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def save_ohlc_data(stock_data, date_column, ticker, interval):
    # Reset the index to make 'Date' or 'Datetime' a column
    stock_data.reset_index(inplace=True)
    
    # Select only the OHLC columns along with 'Date' or 'Datetime'
    ohlc_data = stock_data[[date_column, 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Remove the ticker part from the column names
    ohlc_data.columns = [col[0] if isinstance(col, tuple) else col for col in ohlc_data.columns]

    # Save the OHLC data to a CSV file with the specified format
    # filename = f"data/{ticker}_{interval}_ohlc.csv"
    filename = f"data/{ticker}_{interval}_new_ohlc.csv"
    ohlc_data.to_csv(filename, index=False)

    print(f"OHLC data has been saved to {filename}")

def get_stock_data_intraday(ticker, interval):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=59)
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    save_ohlc_data(stock_data, 'Datetime', ticker, interval)

def get_stock_data_daily(ticker, period):
    stock_data = yf.download(ticker, period=period, interval='1d')
    save_ohlc_data(stock_data, 'Date', ticker, '1d')

# Define the stock ticker and the period for which you want to download data
ticker = 'NVDA'
intervals = ["15m", "1h", "1d"]

for interval in intervals:
    if interval in ['15m', '1h']:
        get_stock_data_intraday(ticker, interval)
    else:
        get_stock_data_daily(ticker, '1y')