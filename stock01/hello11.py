# -*- coding: utf-8 -*-
#计算涨跌幅
import itertools

items = [1,2,3]

for item in itertools.permutations(items):
    print(item)
    
    
for item in itertools.permutations(items,2):
    print(item)
    
    
import numpy as np

stock_cnt = 200
view_days = 504

stock_day_change = np.random.standard_normal((stock_cnt,view_days))

print(stock_day_change.shape)

print(stock_day_change[0:3,:5])

stock_day_change_four = stock_day_change[:4,:4]

print(stock_day_change_four)

print( '最大涨幅{}'.format(np.max(stock_day_change_four,axis=1)))
print( '最大跌幅{}'.format(np.min(stock_day_change_four,axis=1)))
print( '振幅幅度{}'.format(np.std(stock_day_change_four,axis=1)))
print( '平均涨跌{}'.format(np.mean(stock_day_change_four,axis=1)))

stock_mean = stock_day_change[0].mean()
stock_std = stock_day_change[0].std()

print('股票0 mean 的均值期望:{:.3f}'.format(stock_mean))
print('股票0 std 的振幅标准差:{:.3f}'.format(stock_std))

import matplotlib.pyplot as plt
import scipy.stats as scs

plt.hist(stock_day_change[0],bins=50,normed=True)

fit_linspace = np.linspace(stock_day_change[0].min(),stock_day_change[0].max())
pdf = scs.norm(stock_mean,stock_std).pdf(fit_linspace)

plt.plot(fit_linspace,pdf,lw=2,c='r')


