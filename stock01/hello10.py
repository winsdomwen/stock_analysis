# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tushare as ts

#df = pd.read_csv("600519.csv",index_col='date',parse_dates=['date'])[['open']]

df = pd.read_csv("600519.csv")

'''
open_price = df.head(10)['open'].tolist()
open_price.append(234)
'''

date = df.head(10)['date'].values
open_price = df.head(10)['open'].values

'''
date_array = []
date_base = 20170118

for _ in range(0, len(open_price)):
    date_array.append(str(date_base))
    date_base +=1 


print(date_array)
'''

print(np.mean(open_price),np.var(open_price),np.std(open_price))
print()

stock_tuple_list = [(date,price) for date,price in zip(date,open_price)]
print(stock_tuple_list)

stock_dict =  {date:price for date,price in zip(date,open_price)}
print(stock_dict.keys(),stock_dict.values())

stock_prices_sorted = sorted(stock_dict)

print(stock_prices_sorted[-2])








