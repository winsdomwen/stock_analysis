# -*- coding: utf-8 -*-

from django.http import HttpResponse

import tushare as ts
import json
data = ts.get_hist_data('600848', start='2017-06-05', end='2018-01-09')

column_list = []
for row in data:
    column_list.append(row)
jsonlist = []
for index in range(data[column_list[0]].size):
    dict = {}
    for row in data:
        dict[row] = data[row][index]
    dict['date'] = data.index[index]
    jsonlist.append(dict)

def hello(request):
    return HttpResponse(json.dumps(jsonlist))