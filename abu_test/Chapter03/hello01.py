# -*- coding: utf-8 -*-

import numpy as np

w = np.random.binomial(n=2,p=0.5,size=10)

print(w)

print(sum(np.random.binomial(n=2,p=0.5,size=10000)==2)/10000)