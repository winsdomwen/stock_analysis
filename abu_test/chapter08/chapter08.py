# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

# noinspection PyUnresolvedReferences
import abu_local_env

import abupy
from abupy import AbuFactorBuyBreak
from abupy import AbuFactorSellBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import AbuBenchmark
from abupy import AbuPickTimeWorker
from abupy import AbuCapital
from abupy import AbuKLManager
from abupy import ABuTradeProxy
from abupy import ABuTradeExecute
from abupy import ABuPickTimeExecute
from abupy import AbuMetricsBase
from abupy import ABuMarket
from abupy import AbuPickTimeMaster
from abupy import ABuRegUtil
from abupy import AbuPickRegressAngMinMax
from abupy import AbuPickStockWorker
from abupy import ABuPickStockExecute
from abupy import AbuPickStockPriceMinMax
from abupy import AbuPickStockMaster

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()


"""
    第八章 量化系统——开发

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


def sample_811():
    """
    8.1.1 买入因子的实现
    :return:
    """
    # buy_factors 60日向上突破，42日向上突破两个因子
    buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak},
                   {'xd': 42, 'class': AbuFactorBuyBreak}]
    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)
    kl_pd_manager = AbuKLManager(benchmark, capital)
    # 获取TSLA的交易数据
    kl_pd = kl_pd_manager.get_pick_time_kl_pd('usTSLA')
    abu_worker = AbuPickTimeWorker(capital, kl_pd, benchmark, buy_factors, None)
    abu_worker.fit()

    orders_pd, action_pd, _ = ABuTradeProxy.trade_summary(abu_worker.orders, kl_pd, draw=True)

    ABuTradeExecute.apply_action_to_capital(capital, action_pd, kl_pd_manager)
    capital.capital_pd.capital_blance.plot()
    plt.show()
    
    
if __name__ == "__main__":
    sample_811()    