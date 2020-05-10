# -*- coding: utf-8 -*-
import tushare as ts

ts.set_token('d3b318f02bd8bb3d5c362159a6e461d7089575d27e027b5ac80343d6')
pro = ts.pro_api()

data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

print(data)
