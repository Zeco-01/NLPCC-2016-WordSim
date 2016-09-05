# encoding=UTF-8
"""
Created on 

@author: Administrator
"""

import numpy as np
from scipy.stats.stats import spearmanr, pearsonr


# 函数调用方式评价预测性能，set_1是正确的标注，set_2是预测的结果
def spearman(set_1, set_2):
    return spearmanr(set_1, set_2)


def pearson(set_1,set_2):
    return pearsonr(set_1, set_2)


# 文件方式评价预测性能
def evaluate(fdir, fname, mode='spearman'):
    fr = open('%s/%s' % (fdir, fname), 'r')
    set_gordern, set_pred = [], []
    for line in fr.readlines()[1:]:
        line = line.strip()
        group = line.split('\t')
        set_gordern.append(np.float(group[-2]))
        set_pred.append(np.float(group[-1]))
    result = 'NONE'
    if mode == 'spearman':
        result = spearman(set_gordern, set_pred)
    elif mode == 'pearson':
        result = pearsonr(set_gordern,set_pred)
    return result

if __name__ == '__main__':
    pass
