# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00']
print(pd.to_datetime(datestrs))


from datetime import datetime

dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
          datetime(2011, 1, 7), datetime(2011, 1, 8),
          datetime(2011, 1, 10), datetime(2011, 1, 12)]

ts = pd.Series(np.random.randn(6), index=dates)

print(ts)

longer_ts = pd.Series(np.random.randn(1000),
                       index=pd.date_range('1/1/2000', periods=1000))

print(longer_ts.shape)

index = pd.date_range('2012-04-01', '2012-06-01')

print(index)