# -*- coding: utf-8 -*-

from pyecharts.charts import Calendar
from pyecharts import options as opts
import random
import datetime

import pyecharts

print(pyecharts.__version__)

# 示例数据
begin = datetime.date(2019, 1, 1)
end = datetime.date(2019, 12, 31)
data = [[str(begin + datetime.timedelta(days=i)), random.randint(1000, 25000)]
        for i in range((end - begin).days + 1)]

"""
日历图示例：
"""
calendar = (
        Calendar()
        .add("微信步数", data, calendar_opts=opts.CalendarOpts(range_="2019"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Calendar-基本示例", subtitle="我是副标题"),
            legend_opts=opts.LegendOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts( max_=25000, min_=1000,
                                        orient="horizontal",
                                        is_piecewise=True,
                                        pos_top="230px",
                                        pos_left="100px",
            )
        )
    )

calendar.render_notebook()