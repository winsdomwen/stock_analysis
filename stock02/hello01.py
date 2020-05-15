# -*- coding: utf-8 -*-

import tushare as ts

data = ts.get_today_all()

print(data.shape)

