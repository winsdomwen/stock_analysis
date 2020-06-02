# -*- coding: utf-8 -*-

import tushare as ts

'''
df = ts.get_hist_data('000875')

df.to_csv('000875.csv',columns=['open','high','low','close'])

print(df.head())
'''

stock = ['000825','300022','002536','300459','002400','603318'];


for code in stock:
    df = ts.get_hist_data(code)

    df.to_csv(code+'.csv')