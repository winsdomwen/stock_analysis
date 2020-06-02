# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
from abupy import ABuSymbolPd
from abupy import xrange, pd_resample

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()

stock_day_change = np.load('../gen/stock_day_change.npy')

"""
    第四章 量化工具——pandas

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


def sample_411():
    """
    4.1.1 DataFrame构建及方法
    :return:
    """
    print('stock_day_change.shape:', stock_day_change.shape)

    # 下面三种写法输出完全相同，输出如表4-1所示
    print('head():\n', pd.DataFrame(stock_day_change).head())
    print('head(5):\n', pd.DataFrame(stock_day_change).head(5))
    print('[:5]:\n', pd.DataFrame(stock_day_change)[:5])
    
def sample_412():
    """
    4.1.2 索引行列序列
    :return:
    """
    # 股票0 -> 股票stock_day_change.shape[0]
    stock_symbols = ['股票 ' + str(x) for x in xrange(stock_day_change.shape[0])]
    # 通过构造直接设置index参数，head(2)就显示两行，表4-2所示
    print('pd.DataFrame(stock_day_change, index=stock_symbols).head(2):\n',
          pd.DataFrame(stock_day_change, index=stock_symbols).head(2))
    # 从2017-1-1向上时间递进，单位freq='1d'即1天
    days = pd.date_range('2017-1-1', periods=stock_day_change.shape[1], freq='1d')
    # 股票0 -> 股票stock_day_change.shape[0]
    stock_symbols = ['股票 ' + str(x) for x in xrange(stock_day_change.shape[0])]
    # 分别设置index和columns
    df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)
    # 表4-3所示
    print('df.head(2):\n', df.head(2))
    
def sample_413():
    """
    4.1.3 金融时间序列
    :return:
    """
    days = pd.date_range('2017-1-1',  periods=stock_day_change.shape[1], freq='1d')
    stock_symbols = ['股票 ' + str(x) for x in xrange(stock_day_change.shape[0])]
    df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)

    # df做个转置
    df = df.T
    # 表4-4所示
    print('df.head():\n', df.head())

    df_20 = pd_resample(df, '21D', how='mean')
    # 表4-5所示
    print('df_20.head():\n', df_20.head())


def sample_414():
    """
    4.1.4 Series构建及方法
    :return
    """
    days = pd.date_range('2017-1-1',
                         periods=stock_day_change.shape[1], freq='1d')
    stock_symbols = ['股票 ' + str(x) for x in  xrange(stock_day_change.shape[0])]
    df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)
    df = df.T

    print('df.head():\n', df.head())
    df_stock0 = df['股票 0']
    # 打印df_stock0类型
    print('type(df_stock0):', type(df_stock0))
    # 打印出Series的前5行数据, 与DataFrame一致
    print('df_stock0.head():\n', df_stock0.head())

    df_stock0.cumsum().plot()
    plt.show()
    
def sample_415():
    """
    4.1.5 重采样数据
    :return
    """
    days = pd.date_range('2017-1-1',
                         periods=stock_day_change.shape[1], freq='1d')
    stock_symbols = ['股票 ' + str(x) for x in
                     xrange(stock_day_change.shape[0])]
    df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)
    df = df.T
    df_stock0 = df['股票 0']

    # 以5天为周期重采样（周k）
    df_stock0_5 = pd_resample(df_stock0.cumsum(), '5D', how='ohlc')
    # 以21天为周期重采样（月k），
    # noinspection PyUnusedLocal
    df_stock0_20 = pd_resample(df_stock0.cumsum(), '21D', how='ohlc')
    # 打印5天重采样，如下输出2017-01-01, 2017-01-06, 2017-01-11, 表4-6所示
    print('df_stock0_5.head():\n', df_stock0_5.head())

    from abupy import ABuMarketDrawing
    # 图4-2所示
    ABuMarketDrawing.plot_candle_stick(df_stock0_5.index,
                                       df_stock0_5['open'].values,
                                       df_stock0_5['high'].values,
                                       df_stock0_5['low'].values,
                                       df_stock0_5['close'].values,
                                       np.random.random(len(df_stock0_5)),
                                       None, 'stock', day_sum=False,
                                       html_bk=False, save=False)

    print('type(df_stock0_5.open.values):', type(df_stock0_5['open'].values))
    print('df_stock0_5.open.index:\n', df_stock0_5['open'].index)
    print('df_stock0_5.columns:\n', df_stock0_5.columns)



"""
    4.2 基本数据分析示例
"""
# n_folds=2两年
tsla_df = ABuSymbolPd.make_kl_df('usTSLA', n_folds=2)


def sample_420():
    # 表4-7所示
    print('tsla_df.tail():\n', tsla_df.tail())


def sample_421():
    """
    4.2.1 数据整体分析
    :return:
    """
    print('tsla_df.info():\n', tsla_df.info())
    print('tsla_df.describe():\n', tsla_df.describe())

    tsla_df[['close', 'volume']].plot(subplots=True, style=['r', 'g'], grid=True)
    plt.show()
    
def sample_422():
    """
    4.2.2 索引选取和切片选择
    :return:
    """

    # 2014-07-23至2014-07-31 开盘价格序列
    print('tsla_df.loc[x:x, x]\n', tsla_df.loc['2014-07-23':'2014-07-31', 'open'])

    # 2014-07-23至2014-07-31 所有序列，表4-9所示
    print('tsla_df.loc[x:x]\n', tsla_df.loc['2014-07-23':'2014-07-31'])

    # [1:5]：(1，2，3，4)，[2:6]: (2, 3, 4, 5)
    # 表4-10所示
    print('tsla_df.iloc[1:5, 2:6]:\n', tsla_df.iloc[1:5, 2:6])

    # 切取所有行[2:6]: (2, 3, 4, 5)列
    print('tsla_df.iloc[:, 2:6]:\n', tsla_df.iloc[:, 2:6])
    # 选取所有的列[35:37]:(35, 36)行，表4-11所示
    print('tsla_df.iloc[35:37]:\n', tsla_df.iloc[35:37])

    # 指定一个列
    print('tsla_df.close[0:3]:\n', tsla_df.close[0:3])
    # 通过组成一个列表选择多个列，表4-12所示
    print('tsla_df[][0:3]:\n', tsla_df[['close', 'high', 'low']][0:3])
    
def sample_423():
    """
    4.2.3 逻辑条件进行数据筛选
    :return:
    """
    # abs为取绝对值的意思，不是防抱死，表4-13所示
    print('tsla_df[np.abs(tsla_df.p_change) > 8]:\n', tsla_df[np.abs(tsla_df.p_change) > 8])
    print('tsla_df[(np.abs(tsla_df.p_change) > 8) & (tsla_df.volume > 2.5 * tsla_df.volume.mean())]:\n',
          tsla_df[(np.abs(tsla_df.p_change) > 8) & (tsla_df.volume > 2.5 * tsla_df.volume.mean())])


def sample_431():
    """
    4.3.1 数据的离散化
    :return:
    """
    tsla_df.p_change.hist(bins=80)
    plt.show()

    cats = pd.qcut(np.abs(tsla_df.p_change), 10)
    print('cats.value_counts():\n', cats.value_counts())

    # 将涨跌幅数据手工分类，从负无穷到－7，－5，－3，0， 3， 5， 7，正无穷
    bins = [-np.inf, -7.0, -5, -3, 0, 3, 5, 7, np.inf]
    cats = pd.cut(tsla_df.p_change, bins)
    print('bins cats.value_counts():\n', cats.value_counts())

    # cr_dummies为列名称前缀
    change_ration_dummies = pd.get_dummies(cats, prefix='cr_dummies')
    print('change_ration_dummies.head():\n', change_ration_dummies.head())


if __name__ == "__main__":
    sample_431()    
