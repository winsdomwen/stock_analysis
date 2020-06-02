# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
from abupy import ABuSymbolPd
from abupy import pd_rolling_std, pd_ewm_std, pd_rolling_mean

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()

tsla_df = ABuSymbolPd.make_kl_df('usTSLA', n_folds=2)

"""
    第五章 量化工具——可视化

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


# noinspection PyUnresolvedReferences
def plot_demo(axs=None, just_series=False):
    """
    绘制tsla的收盘价格曲线
    :param axs: axs为子画布，稍后会详细讲解
    :param just_series: 是否只绘制一条收盘曲线使用series，后面会用到
    :return:
    """
    # 如果参数传入子画布则使用子画布绘制，下一节会使用
    drawer = plt if axs is None else axs
    # Series对象tsla_df.close，红色
    drawer.plot(tsla_df.close, c='r')
    if not just_series:
        # 为曲线不重叠，y变量加了10个单位tsla_df.close.values + 10
        # numpy对象tsla_df.close.index ＋ tsla_df.close.values，绿色
        drawer.plot(tsla_df.close.index, tsla_df.close.values + 10,
                    c='g')
        # 为曲线不重叠，y变量加了20个单位
        # list对象，numpy.tolist()将numpy对象转换为list对象，蓝色
        drawer.plot(tsla_df.close.index.tolist(),
                    (tsla_df.close.values + 20).tolist(), c='b')

    plt.xlabel('time')
    plt.ylabel('close')
    plt.title('TSLA CLOSE')
    plt.grid(True)
    
def sample_511():
    """
    5.1.1 matplotlib可视化基础
    :return:
    """
    print('tsla_df.tail():\n', tsla_df.tail())

    plot_demo()
    plt.show()
    
def sample_512():
    """
    5.1.2 matplotlib子画布及loc的使用
    :return:
    """
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    # 画布0，loc：0 plot_demo中传入画布，则使用传入的画布绘制
    drawer = axs[0][0]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], loc=0)
    # 画布1，loc：1
    drawer = axs[0][1]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], loc=1)
    # 画布2，loc：2
    drawer = axs[1][0]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], loc=2)
    # 画布3，loc：2， 设置bbox_to_anchor，在画布外的相对位置绘制
    drawer = axs[1][1]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], bbox_to_anchor=(1.05, 1),
                  loc=2, borderaxespad=0.)
    plt.show()


def sample_513():
    """
    5.1.3 k线图的绘制
    :return:
    """
    import matplotlib.finance as mpf

    __colorup__ = "red"
    __colordown__ = "green"
    # 为了示例清晰，只拿出前30天的交易数据绘制蜡烛图，
    tsla_part_df = tsla_df[:30]
    fig, ax = plt.subplots(figsize=(14, 7))
    qutotes = []

    for index, (d, o, c, h, l) in enumerate(
            zip(tsla_part_df.index, tsla_part_df.open, tsla_part_df.close,
                tsla_part_df.high, tsla_part_df.low)):
        # 蜡烛图的日期要使用matplotlib.finance.date2num进行转换为特有的数字值
        d = mpf.date2num(d)
        # 日期，开盘，收盘，最高，最低组成tuple对象val
        val = (d, o, c, h, l)
        # 加val加入qutotes
        qutotes.append(val)
    # 使用mpf.candlestick_ochl进行蜡烛绘制，ochl代表：open，close，high，low
    mpf.candlestick_ochl(ax, qutotes, width=0.6, colorup=__colorup__,
                         colordown=__colordown__)
    ax.autoscale_view()
    ax.xaxis_date()
    plt.show()

def sample_52():
    """
    5.2 使用bokeh交互可视化
    :return:
    """
    from abupy import ABuMarketDrawing
    ABuMarketDrawing.plot_candle_form_klpd(tsla_df, html_bk=True)
    
"""
    5.3 使用pandas可视化数据
"""


def sample_531_1():
    """
    5.3.1_1 绘制股票的收益，及收益波动情况 demo list
    :return:
    """
    # 示例序列
    demo_list = np.array([2, 4, 16, 20])
    # 以三天为周期计算波动
    demo_window = 3
    # pd.rolling_std * np.sqrt
    print('pd.rolling_std(demo_list, window=demo_window, center=False) * np.sqrt(demo_window):\n',
          pd_rolling_std(demo_list, window=demo_window, center=False) * np.sqrt(demo_window))

    print('pd.Series([2, 4, 16]).std() * np.sqrt(demo_window):', pd.Series([2, 4, 16]).std() * np.sqrt(demo_window))
    print('pd.Series([4, 16, 20]).std() * np.sqrt(demo_window):', pd.Series([4, 16, 20]).std() * np.sqrt(demo_window))
    print('np.sqrt(pd.Series([2, 4, 16]).var() * demo_window):', np.sqrt(pd.Series([2, 4, 16]).var() * demo_window))


def sample_531_2():
    """
    5.3.1_2 绘制股票的收益，及收益波动情况
    :return:
    """
    tsla_df_copy = tsla_df.copy()
    # 投资回报
    tsla_df_copy['return'] = np.log(tsla_df['close'] / tsla_df['close'].shift(1))

    # 移动收益标准差
    tsla_df_copy['mov_std'] = pd_rolling_std(tsla_df_copy['return'],
                                             window=20,
                                             center=False) * np.sqrt(20)
    # 加权移动收益标准差，与移动收益标准差基本相同，只不过根据时间权重计算std
    tsla_df_copy['std_ewm'] = pd_ewm_std(tsla_df_copy['return'], span=20,
                                         min_periods=20,
                                         adjust=True) * np.sqrt(20)

    tsla_df_copy[['close', 'mov_std', 'std_ewm', 'return']].plot(subplots=True, grid=True)
    plt.show()


def sample_532():
    """
    5.3.2 绘制股票的价格与均线
    :return:
    """
    tsla_df.close.plot()
    # ma 30
    # pd_rolling_mean(tsla_df.close, window=30).plot()
    pd_rolling_mean(tsla_df.close, window=30).plot()
    # ma 60
    # pd.rolling_mean(tsla_df.close, window=60).plot()
    pd_rolling_mean(tsla_df.close, window=60).plot()
    # ma 90
    # pd.rolling_mean(tsla_df.close, window=90).plot()
    pd_rolling_mean(tsla_df.close, window=90).plot()
    # loc='best'即自动寻找适合的位置
    plt.legend(['close', '30 mv', '60 mv', '90 mv'], loc='best')
    plt.show()
    
def sample_533():
    """
    5.3.3 其它pandas统计图形种类
    :return:
    """
    # iloc获取所有低开高走的下一个交易日组成low_to_high_df，由于是下一个交易日
    # 所以要对满足条件的交易日再次通过iloc获取，下一个交易日index用key.values + 1
    # key序列的值即为0-len(tsla_df), 即为交易日index，详情查阅本章初tail
    low_to_high_df = tsla_df.iloc[tsla_df[
                                      (tsla_df.close > tsla_df.open) & (
                                          tsla_df.key != tsla_df.shape[
                                              0] - 1)].key.values + 1]

    # 通过where将下一个交易日的涨跌幅通过ceil，floor向上，向下取整
    change_ceil_floor = np.where(low_to_high_df['p_change'] > 0,
                                 np.ceil(
                                     low_to_high_df['p_change']),
                                 np.floor(
                                     low_to_high_df['p_change']))

    # 使用pd.Series包裹，方便之后绘制
    change_ceil_floor = pd.Series(change_ceil_floor)
    print('低开高收的下一个交易日所有下跌的跌幅取整和sum: ' + str(
        change_ceil_floor[change_ceil_floor < 0].sum()))

    print('低开高收的下一个交易日所有上涨的涨幅取整和sum: ' + str(
        change_ceil_floor[change_ceil_floor > 0].sum()))

    # 2 * 2: 四张子图
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    # 竖直柱状图，可以看到-1的柱子最高, 图5-7左上
    change_ceil_floor.value_counts().plot(kind='bar', ax=axs[0][0])
    # 水平柱状图，可以看到-1的柱子最长, 图5-7右上
    change_ceil_floor.value_counts().plot(kind='barh', ax=axs[0][1])
    # 概率密度图，可以看到向左偏移, 图5-7左下
    change_ceil_floor.value_counts().plot(kind='kde', ax=axs[1][0])
    # 圆饼图，可以看到－1所占的比例最高, -2的比例也大于＋2，图5-7右下
    change_ceil_floor.value_counts().plot(kind='pie', ax=axs[1][1])
    plt.show()

def sample_54_1():
    """
    5.4 使用seaborn可视化数据
    :return:
    """
    sns.distplot(tsla_df['p_change'], bins=80)
    plt.show()

    sns.boxplot(x='date_week', y='p_change', data=tsla_df)
    plt.show()

    sns.jointplot(tsla_df['high'], tsla_df['low'])
    plt.show()
    
def sample_54_2():
    """
    5.4 使用seaborn可视化数据
    :return:
    """
    change_df = pd.DataFrame({'tsla': tsla_df.p_change})
    # join usGOOG
    change_df = change_df.join(pd.DataFrame({'goog': ABuSymbolPd.make_kl_df('usGOOG', n_folds=2).p_change}),
                               how='outer')
    # join usAAPL
    change_df = change_df.join(pd.DataFrame({'aapl': ABuSymbolPd.make_kl_df('usAAPL', n_folds=2).p_change}),
                               how='outer')
    # join usFB
    change_df = change_df.join(pd.DataFrame({'fb': ABuSymbolPd.make_kl_df('usFB', n_folds=2).p_change}),
                               how='outer')
    # join usBIDU
    change_df = change_df.join(pd.DataFrame({'bidu': ABuSymbolPd.make_kl_df('usBIDU', n_folds=2).p_change}),
                               how='outer')

    change_df = change_df.dropna()
    # 表5-2所示
    print('change_df.head():\n', change_df.head())

    # 使用corr计算数据的相关性
    corr = change_df.corr()
    _, ax = plt.subplots(figsize=(8, 5))
    # sns.heatmap热力图展示每组股票涨跌幅的相关性
    sns.heatmap(corr, ax=ax)
    plt.show()


def sample_55_1():
    """
    5.5 可视化量化策略的交易区间，卖出原因
    :return:
    """

    def plot_trade(buy_date, sell_date):
        # 找出2014-07-28对应时间序列中的index作为start
        start = tsla_df[tsla_df.index == buy_date].key.values[0]
        # 找出2014-09-05对应时间序列中的index作为end
        end = tsla_df[tsla_df.index == sell_date].key.values[0]

        # 使用5.1.1封装的绘制tsla收盘价格时间序列函数plot_demo
        # just_series＝True, 即只绘制一条曲线使用series数据
        plot_demo(just_series=True)

        # 将整个时间序列都填充一个底色blue，注意透明度alpha=0.08是为了
        # 之后标注其他区间透明度高于0.08就可以清楚显示
        plt.fill_between(tsla_df.index, 0, tsla_df['close'], color='blue',
                         alpha=.08)

        # 标注股票持有周期绿色，使用start和end切片周期
        # 透明度alpha=0.38 > 0.08
        plt.fill_between(tsla_df.index[start:end], 0,
                         tsla_df['close'][start:end], color='green',
                         alpha=.38)

        # 设置y轴的显示范围，如果不设置ylim，将从0开始作为起点显示，效果不好
        plt.ylim(np.min(tsla_df['close']) - 5,
                 np.max(tsla_df['close']) + 5)
        # 使用loc='best'
        plt.legend(['close'], loc='best')

    # 标注交易区间2014-07-28到2014-09-05, 图5-12所示
    plot_trade('2014-07-28', '2014-09-05')
    plt.show()

    def plot_trade_with_annotate(buy_date, sell_date, annotate):
        """
        :param buy_date: 交易买入日期
        :param sell_date: 交易卖出日期
        :param annotate: 卖出原因
        :return:
        """
        # 标注交易区间buy_date到sell_date
        plot_trade(buy_date, sell_date)
        # annotate文字，asof：从tsla_df['close']中找到index:sell_date对应值
        plt.annotate(annotate,
                     xy=(sell_date, tsla_df['close'].asof(sell_date)),
                     arrowprops=dict(facecolor='yellow'),
                     horizontalalignment='left', verticalalignment='top')

    plot_trade_with_annotate('2014-07-28', '2014-09-05',
                             'sell for stop loss')
    plt.show()

if __name__ == "__main__":
    sample_55_1()
