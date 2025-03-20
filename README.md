# Stocks GADF Project

This project downloads OHLC (Open, High, Low, Close) stock data using the `yfinance` library and generates Gramian Angular Field (GAF) images from the stock's closing prices.

## Requirements

- Python 3.x
- pandas
- yfinance
- mplfinance
- matplotlib
- pyts

You can install the required packages using pip:

```sh
pip install pandas yfinance mplfinance matplotlib pyts
```

## Usage

### Download OHLC Data

The script `get_ohlc.py` downloads OHLC data for a specified stock ticker and intervals. The data is saved as CSV files in the `data` directory.

To run the script:

```sh
python get_ohlc.py
```

### Generate GAF Images

The Jupyter notebook `gadf.ipynb` loads the OHLC data, plots the candlestick chart, and generates GAF images from the closing prices.

To run the notebook:

1. Open `gadf.ipynb` in Jupyter Notebook or JupyterLab.
2. Execute the cells to load the data, plot the candlestick chart, and generate the GAF images.

## File Structure

- `get_ohlc.py`: Script to download OHLC data and save it as CSV files.
- `gadf.ipynb`: Jupyter notebook to load OHLC data, plot candlestick charts, and generate GAF images.
- `data/`: Directory where the OHLC CSV files are saved.
- `images/`: Directory where the GAF images are saved.

## Example

To download OHLC data for the stock ticker `LUNR` with intervals `15m`, `1h`, and `1d`, and generate GAF images:

1. Run `get_ohlc.py` to download the data.
2. Open `gadf.ipynb` and execute the cells to generate the GAF images.

The OHLC data will be saved in the `data` directory, and the GAF images will be saved in the `images` directory.
