# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import time
#获取数据


df = pd.read_csv("600519.csv")[['date','open','close','high','low','volume']]

df=df[['date','open','high','low','close','volume']]

print(df.head())


def get_EMA(df,N):
    for i in range(len(df)):
        if i==0:
            df.ix[i,'ema']=df.ix[i,'close']
#            df.ix[i,'ema']=0
        if i>0:
            df.ix[i,'ema']=(2*df.ix[i,'close']+(N-1)*df.ix[i-1,'ema'])/(N+1)
    ema=list(df['ema'])
    return ema
 
def get_MACD(df,short=12,long=26,M=9):
    a=get_EMA(df,short)
    b=get_EMA(df,long)
    df['diff']=pd.Series(a)-pd.Series(b)
    #print(df['diff'])
    for i in range(len(df)):
        if i==0:
            df.ix[i,'dea']=df.ix[i,'diff']
        if i>0:
            df.ix[i,'dea']=((M-1)*df.ix[i-1,'dea']+2*df.ix[i,'diff'])/(M+1)
    df['macd']=2*(df['diff']-df['dea'])
    return df

print(get_MACD(df,12,26,9))
