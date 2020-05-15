# -*- coding: utf-8 -*-

import tushare as ts
df =ts.get_hist_data('600848')

df.to_csv("600848.csv")