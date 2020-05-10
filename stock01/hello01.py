# -*- coding: utf-8 -*-

# 导入包
import pandas as pd
import pandas.io.data as web # Package and modules for importing data; this code may change depending on pandas version
import datetime

# We will look at stock prices over the past year, starting at January 1, 2016
start = datetime.datetime(2016,1,1)
end = datetime.date.today()

# Let's get Apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
apple = web.DataReader("AAPL", "yahoo", start, end)

print(type(apple))


