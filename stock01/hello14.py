# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:19:00 2020

@author: Administrator
"""

import mplfinance as mpf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("600519.csv",index_col='date',parse_dates=['date'])[['open','close','high','low']]

#mpf.plot(df, type = 'candlestick')

#print(df.close)

moving_avg = df.close.rolling(5).mean()

#pd.rolling_mean(df.close,window=5).plot()

plt.plot(df.close, color='red') 
plt.plot(moving_avg, color='green') 