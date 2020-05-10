import pandas as pd
import numpy as np

def load_today_all():
    '''
    [加载当日实时数据]######
    从硬盘中获取当日交易的数据，数据由update_today_all提供更新
    【行情不含基金和ETF】
    读取目录在/data/today_all.csv
    '''
    allcode=[]
    #载入代码
    df=pd.read_csv('stock.csv')
    #筛选代码
    df.set_index(['code'], inplace = True) 
    for i in df.index:
        i = "%06d" % i
        #i=i.zfill(6)
        allcode.append(i)
        print(i)
    return allcode

#allcode = load_today_all()


#print(",".join(str(i) for i in allcode))

'''
f = open('stock_code.txt', 'w',encoding = 'utf8')
f.write(",".join(str(i) for i in allcode))
'''

fname = 'stock_code.txt'

with open(fname, 'r+', encoding='utf-8') as f:
    s = [i[:-1].split(',') for i in f.readlines()]
    print(np.squeeze(np.array(s)).shape)