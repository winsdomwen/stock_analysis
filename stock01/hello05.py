# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
 
#df = ts.get_k_data("600519",start="2018-01-01")
 
#df.to_csv("600519.csv")
 
df = pd.read_csv("600519.csv",index_col='date',parse_dates=['date'])[['open','close','high','low']]
 
print(df[(df['close']-df['open'])/df['open']>=0.03].index)

print(df[(df['open']-df['close'].shift(1))/df['close'].shift(1)<=-0.02].index)

