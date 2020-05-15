# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
 
df = pd.read_csv("600519.csv",index_col='date',parse_dates=['date'])[['open','close','low','high']]

df['ma5'] = np.nan #新加列
df['ma30'] = np.nan
#循环写法
for i in range(4,len(df)): #从第5行开始到最后
    df.loc[df.index[i],'ma5'] = df['close'][i-4:i+1].mean() #每5个求平均并赋值
 
for i in range(29,len(df)):
    df.loc[df.index[i],'ma30'] = df['close'][i-29:i+1].mean()
 
#rolling写法,滚动数据取值
df['ma5'] = df['close'].rolling(5).mean()
df['ma30'] = df['close'].rolling(10).mean()

print(df['ma30'])

