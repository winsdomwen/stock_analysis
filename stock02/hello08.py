# -*- coding: utf-8 -*-
import time
import datetime
import random

import tushare as ts
import pandas as pd

stock_list_file = 'stock_list.csv'   

tushare_token='d3b318f02bd8bb3d5c362159a6e461d7089575d27e027b5ac80343d6'

ts.set_token(tushare_token)

pro = ts.pro_api()

df = ts.pro_bar(ts_code='000001.SZ', adj='qfq', start_date='20180101', end_date='20181011')

print(df.head(10))