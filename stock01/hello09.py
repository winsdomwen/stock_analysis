import pandas as pd
import numpy as np
import tushare as ts
 
df = ts.get_k_data("600519", start="2020-01-01")
df.to_csv("600519.csv")
 
df = pd.read_csv("600519.csv",index_col='date',parse_dates=['date'])[['open','close','high','low']]

print(df.head(10))

print(df[(df['close']-df['open'])/df['open']>0.03].index)

df_monthly = df.resample('M').first()
df_yearly = df.resample('A').last()[:-1]
cost_money = 0
hold = 0
for year in range(2019, 2020):
    cost_money += df_monthly[str(year)]['open'].sum()*100
    hold += len(df_monthly[str(year)]['open']) * 100
    if year != 2018:
        cost_money -= df_yearly[str(year)]['open'][0] * hold
        hold = 0
    print(cost_money)
     
cost_money -= hold * price_last
 
print(-cost_money)