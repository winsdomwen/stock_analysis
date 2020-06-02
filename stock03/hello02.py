# -*- coding: utf-8 -*-

#画K线图
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from matplotlib.pylab import date2num
import datetime
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

data_list = []

'''
df = pd.read_csv("000825.csv",index_col='date',parse_dates=['date'])[['open','close','high','low']]

for dates,row in df.iterrows():
    # 将时间转换为数字
    t = date2num(dates)
    open,high,low,close = row[:4]
    datas = (t,open,high,low,close)
    data_list.append(datas)
    
    
# 创建子图
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
# 设置X轴刻度为日期时间
ax.xaxis_date()
plt.xticks(rotation=45)
plt.yticks()
plt.title(u"股票代码：601558两年K线图")
plt.xlabel(u"时间")
plt.ylabel(u"股价（元）")
mpf.candlestick_ohlc(ax,data_list,width=1.5,colorup='r',colordown='green')
plt.grid()

'''

'''
df = pd.read_csv("000825.csv")[['date','open','close','high','low']]
df = df.loc[0:100]

df = df.set_index('date')


for dates,row in df.iterrows():
    # 将时间转换为数字

    open = row['open']
    high = row['high']
    low = row['low']
    close = row['close']
     

    datas = (dates,open,high,low,close)
    data_list.append(datas)


print(df['open'].std())
'''

df = pd.read_csv("000825.csv",index_col='date',parse_dates=['date'])[['open','close','high','low']]

#print(df.head(10))
df  = df.loc['2020-05-29':'2020-01-02']


result = df.sort_index(by='open')[:5]

print(result)

result = df.sort_index(by='open',ascending=False)[:5]
print(result)

print(df.open.max(),df.close.max())

print(df.open.cumsum())