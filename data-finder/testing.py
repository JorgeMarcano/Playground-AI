import yfinance as yf
import numpy
print("Done importing")

data = yf.download("AAPL", start="2020-04-21", end="2020-04-28", interval="1d")
#days = yf.download("AAPL", period="5d")

#dates = data['Open'].index

#print(days.index[0])

#print(days['Open'].at_time(dates[0].normalize().time()))


print(data.index)
