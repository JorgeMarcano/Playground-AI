import yfinance as yf
import numpy
print("Done importing")

stocks = ["AAPL", "GOOGL", "MCD", "WMT", "IGA", "SBUX", "BBY", "URBN", "HD", "NFLX", "AMZN", "FB", "EBAY", "FDX", "F", "GM", "GE", "AMD", "INTC", "IBM"]

for i in stock:

    data = yf.download("TSX:CTC.A", start="2020-04-21", end="2020-04-28", interval="1d")

    for i in data:
        if data[i].isnull().values.any():
            print(data[i])
