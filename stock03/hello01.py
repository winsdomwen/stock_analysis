# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
#import tushare as ts

 
df = pd.read_csv("000825.csv",index_col='date',parse_dates=['date'])[['open','close','high','low']]


#print(df['open'])
days = pd.date_range('2017-11-29', periods=df.shape[0], freq='1d')

#df['date'] = pd.to_datetime(df['date'])

#df = pd.read_csv("000825.csv")[['date','open','close','high','low']]

df['open'].plot()
df['close'].plot()

df2 = pd.read_csv("002400.csv",index_col='date',parse_dates=['date'])[['open','close','high','low']]

df2['open'].plot()
df2['close'].plot()