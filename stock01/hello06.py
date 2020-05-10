# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts

df = pd.read_csv("600519.csv",index_col='date',parse_dates=['date'])[['open','close','high','low']]


df['ma5'] = np.nan #新加列
df['ma30'] = np.nan
#循环写法
for i in range(4,len(df)): #从第5行开始到最后
    df.loc[df.index[i],'ma5'] = df['close'][i-4:i+1].mean() #每5个求平均并赋值
 
for i in range(29,len(df)):
    df.loc[df.index[i],'ma30'] = df['close'][i-29:i+1].mean()
 
#rolling写法,滚动数据取值
df['ma5'] = df['close'].rolling(5).mean()
df['ma30'] = df['close'].rolling(30).mean()

df[['close','ma5','ma30']].plot()
 
plt.show()
 
df.loc['2019',['close','ma5','ma30']].plot() #只画一年
 
plt.show()

df = df.dropna()
df = df['2010-01-01':] #截取数据
golden_cross = []
death_cross = []
#循环写法
for i in range(30,len(df)):#按天循环,从长期均线有值的数据开始
    #今天的ma5>=ma30并且昨天的ma5小于ma30
    if df['ma5'][i] >= df['ma30'][i] and df['ma5'][i-1]<df['ma30'][i-1]:
        golden_cross.append(df.index[i].to_pydatetime())#保存金叉数据
    #今天的ma5<=ma30并且昨天的ma5大于ma30
    if df['ma5'][i] <= df['ma30'][i] and df['ma5'][i-1]>df['ma30'][i-1]:
        death_cross.append(df.index[i].to_pydatetime())#保存死叉数据
 
#shift写法，不使用循环
str1 = df['ma5'] < df['ma30']
str2 = df['ma5'] >= df['ma30']
 
death_cross = df[str1 & str2.shift(1)].index
golden_cross = df[~(str1 | str2.shift(1))].index

first_money = 100000 #初始资金
money = first_money
hold = 0 #持有股份数量
#日期排序
sr1 = pd.Series(1,index=golden_cross)
sr2 = pd.Series(0,index=death_cross)
sr = sr1.append(sr2).sort_index()
 
for i in range(0,len(sr)):
    p = df['open'][sr.index[i]] #当天的开盘价格
    #金叉
    if sr.iloc[i] == 1:
        buy = (money // (100*p)) #整除（按手计算）能买多少
        hold += buy * 100
        money -= buy*100*p
    else:
        money += hold * p
        hold = 0
 
p = df['open'][-1] #最后一天开盘价
now_money = hold * p + money
print(now_money - first_money)