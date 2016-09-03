#!/user/bin/python

import bhtsne
import input_data
import numpy as np
import pandas as pd
import pylab as py
import timeit
import os

partition = 50
rand_seed = 12345
perplexity = 30
no_dims = 3
file_name = input_data.get_file_name('20160830')
with open(file_name, 'r') as content_file:
    data = content_file.read().split('\n')
X = [s[0:s.rfind(',')].replace(',', '\t') for s in data]
Y = [s[s.rfind(',')+1:] for s in data]
data = None
del X[0]
del Y[0]
del X[-1]
del Y[-1]
Y = [int(s) for s in Y]

for pp in range(5, 51, 5):
    reducedX = bhtsne.run_bh_tsne(X[::partition], randseed=rand_seed, perplexity=pp, no_dims=no_dims)
    out = pd.DataFrame(reducedX) 
    out['target'] = Y[::partition]
    out.to_csv(file_name.replace('.csv', '_p%d_pp%d.out' % (partition, pp)), index=False)