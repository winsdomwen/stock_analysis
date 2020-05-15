# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
 
df = pd.read_csv("600519.csv",index_col='date',parse_dates=['date'])[['open','close','low','high']]


pro = ts.pro_api()

