import yfinance as yf
import pandas as pd

def get_data(ticker = "MSFT", interval = "30m", period="5wk"):
    data = yf.download(ticker, period=period, interval=interval)
    data.columns = [' '.join(col).strip() for col in data.columns]
    data.index = data.index.tz_localize(None)

    if not data.empty:
        data = data.rename(columns={'Open AAPL': 'open', 'High AAPL': 'high', 'Low AAPL': 'low', 'Close AAPL': 'close', 'Volume AAPL': 'volume'})
        data = data.reset_index()
        data.to_csv('stocks_mfst_daily.csv', index=False)
    
        #print(data)
        return data
    else:
        print("Could not retrieve data from yfinance.")
        return None

data = get_data(ticker = "MSFT", interval = "1d", period="7y")