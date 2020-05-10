# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:39:49 2020

@author: Administrator
"""
import tushare as ts

df = ts.get_stock_basics()

df.to_csv("stock.csv")
