# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections import namedtuple
import itertools
# noinspection PyCompatibility
from concurrent.futures import ProcessPoolExecutor
# noinspection PyCompatibility
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
from abupy import six, xrange, range, reduce, map, filter, partial
from abupy import ABuSymbolPd

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()


"""
    第二章 量化语言——Python

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


def sample_211():
    """
    量化语言-Python
    :return:
    """
    price_str = '30.14, 29.58, 26.36, 32.56, 32.82'
    print('type(price_str):', type(price_str))

    if not isinstance(price_str, str):
        # not代表逻辑‘非’， 如果不是字符串，转换为字符串
        price_str = str(price_str)
    if isinstance(price_str, int) and price_str > 0:
        # and 代表逻辑‘与’，如果是int类型且是正数
        price_str += 1
    elif isinstance(price_str, float) or float(price_str[:4]) < 0:
        # or 代表逻辑‘或’，如果是float或者小于0
        price_str += 1.0
    else:
        try:
            raise TypeError('price_str is str type!')
        except TypeError:
            print('raise, try except')
            
   
def sample_212(show=True):
    """
    2.1.2 字符串和容器
    :return:
    """
    show_func = print if show else lambda a: a
    price_str = '30.14, 29.58, 26.36, 32.56, 32.82'
    show_func('旧的price_str id= {}'.format(id(price_str)))
    price_str = price_str.replace(' ', '')
    show_func('新的price_str id= {}'.format(id(price_str)))
    show_func(price_str)
    # split以逗号分割字符串，返回数组price_array
    price_array = price_str.split(',')
    show_func(price_array)
    # price_array尾部append一个重复的32.82
    price_array.append('32.82')
    show_func(price_array)
    show_func(set(price_array))
    price_array.remove('32.82')
    show_func(price_array)

    date_array = []
    date_base = 20170118
    # 这里用for只是为了计数，无用的变量python建议使用'_'声明
    for _ in xrange(0, len(price_array)):
        date_array.append(str(date_base))
        # 本节只是简单示例，不考虑日期的进位
        date_base += 1
    show_func(date_array)

    date_base = 20170118
    date_array = [str(date_base + ind) for ind, _ in enumerate(price_array)]
    show_func(date_array)

    stock_tuple_list = [(date, price) for date, price in zip(date_array, price_array)]
    # tuple访问使用索引
    show_func('20170119日价格：{}'.format(stock_tuple_list[1][1]))
    show_func(stock_tuple_list)

    stock_namedtuple = namedtuple('stock', ('date', 'price'))
    stock_namedtuple_list = [stock_namedtuple(date, price) for date, price in zip(date_array, price_array)]
    # namedtuple访问使用price
    show_func('20170119日价格：{}'.format(stock_namedtuple_list[1].price))
    show_func(stock_namedtuple_list)

    # 字典推导式：{key: value for in}
    stock_dict = {date: price for date, price in zip(date_array, price_array)}
    show_func('20170119日价格：{}'.format(stock_dict['20170119']))
    show_func(stock_dict)

    show_func(stock_dict.keys())

    stock_dict = OrderedDict((date, price) for date, price in zip(date_array, price_array))
    show_func(stock_dict.keys())
    return stock_dict
         
def sample_221():
    """
    2.2.1 函数的使用和定义
    :return:
    """
    stock_dict = sample_212(show=False)
    print('min(stock_dict):', min(stock_dict))
    print('min(zip(stock_dict.values(), stock_dict.keys())):', min(zip(stock_dict.values(), stock_dict.keys())))

    def find_second_max(dict_array):
        # 对传入的dict sorted排序
        stock_prices_sorted = sorted(zip(dict_array.values(), dict_array.keys()))
        # 第二大的也就是倒数第二个
        return stock_prices_sorted[-2]

    # 系统函数callable验证是否为一个可call的函数
    if callable(find_second_max):
        print('find_second_max(stock_dict):', find_second_max(stock_dict))

if __name__ == "__main__":
    sample_221()