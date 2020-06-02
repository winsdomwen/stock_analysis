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
from abupy import six, xrange

from abc import ABCMeta, abstractmethod


warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()

tsla_close = ABuSymbolPd.make_kl_df('usTSLA').close
# x序列: 0，1，2, ...len(tsla_close)
x = np.arange(0, tsla_close.shape[0])
# 收盘价格序列
y = tsla_close.values


"""
    第六章 量化工具——数学：你一生的追求到底能带来多少幸福

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


def sample_611_1(show=True):
    """
    6.1.1 线性回归
    :return:
    """
    import statsmodels.api as sm
    from statsmodels import regression

    def regress_y(_y):
        _y = _y
        # x序列: 0，1，2, ...len(y)
        _x = np.arange(0, len(_y))
        _x = sm.add_constant(_x)
        # 使用OLS做拟合
        _model = regression.linear_model.OLS(_y, _x).fit()
        return _model

    model = regress_y(y)
    b = model.params[0]
    k = model.params[1]
    # y = kx + b
    y_fit = k * x + b
    if show:
        plt.plot(x, y)
        plt.plot(x, y_fit, 'r')
        plt.show()
        # summary模型拟合概述，表6-1所示
        print(model.summary())
    return y_fit

# noinspection PyPep8Naming
def sample_611_2():
    """
    6.1.1 线性回归
    :return:
    """
    y_fit = sample_611_1(show=False)

    MAE = sum(np.abs(y - y_fit)) / len(y)
    print('偏差绝对值之和(MAE)={}'.format(MAE))
    MSE = sum(np.square(y - y_fit)) / len(y)
    print('偏差绝对值之和(MSE)={}'.format(MSE))
    RMSE = np.sqrt(sum(np.square(y - y_fit)) / len(y))
    print('偏差绝对值之和(RMSE)={}'.format(RMSE))

    from sklearn import metrics
    print('sklearn偏差绝对值之和(MAE)={}'.format(metrics.mean_absolute_error(y, y_fit)))
    print('sklearn偏差平方(MSE)={}'.format(metrics.mean_squared_error(y, y_fit)))
    print('sklearn偏差平方和开平方(RMSE)={}'.format(np.sqrt(metrics.mean_squared_error(y, y_fit))))

# noinspection PyCallingNonCallable
def sample_612():
    """
    6.1.2 多项式回归
    :return:
    """
    import itertools

    # 生成9个subplots 3*3
    _, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    # 将 3 * 3转换成一个线性list
    axs_list = list(itertools.chain.from_iterable(axs))
    # 1-9次多项式回归
    poly = np.arange(1, 10, 1)
    for p_cnt, ax in zip(poly, axs_list):
        # 使用polynomial.Chebyshev.fit进行多项式拟合
        p = np.polynomial.Chebyshev.fit(x, y, p_cnt)
        # 使用p直接对x序列代人即得到拟合结果序列
        y_fit = p(x)
        # 度量mse值
        from sklearn import metrics
        mse = metrics.mean_squared_error(y, y_fit)
        # 使用拟合次数和mse误差大小设置标题
        ax.set_title('{} poly MSE={}'.format(p_cnt, mse))
        ax.plot(x, y, '', x, y_fit, 'r.')
    plt.show()

def sample_613():
    """
    6.1.3 插值
    :return:
    """
    from scipy.interpolate import interp1d, splrep, splev

    # 示例两种插值计算方式
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    # 线性插值
    linear_interp = interp1d(x, y)
    # axs[0]左边的
    axs[0].set_title('interp1d')
    # 在相同坐标系下，同样的x，插值的y值使r.绘制（红色点）
    axs[0].plot(x, y, '', x, linear_interp(x), 'r.')

    # B-spline插值
    splrep_interp = splrep(x, y)
    # axs[1]右边的
    axs[1].set_title('splrep')
    # #在相同坐标系下，同样的x，插值的y值使g.绘制（绿色点）
    axs[1].plot(x, y, '', x, splev(x, splrep_interp), 'g.')
    plt.show()

"""
    6.2 蒙特卡洛方法与凸优化
    6.2.1 你一生的追求到底能带来多少幸福
"""

# 每个人平均寿命期望是75年，约75*365=27375天
K_INIT_LIVING_DAYS = 27375



if __name__ == "__main__":
    sample_613()