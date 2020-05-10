# -*- coding: utf-8 -*-

import tushare as ts


df = ts.get_hist_data('000875')


df.to_csv('000875.csv',columns=['open','high','low','close'])

print(df.head())