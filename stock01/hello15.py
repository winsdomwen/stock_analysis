# -*- coding: utf-8 -*-

import seaborn as sns
import pandas as pd

df = pd.read_csv("600519.csv",index_col='date',parse_dates=['date'])[['open','close','high','low']]

sns.distplot(df.close,bins=10)