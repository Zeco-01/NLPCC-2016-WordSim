# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
import post
from Com import macro
from Eval import eval
from Proc.LR import sk_LR
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


def add(lst):
    if len(lst) == 0:
        return
    lst[0] += 1

    if lst[0] == 2:
        lst[0] = 0
        lst[1:] = add(lst[1:])
    return lst


def get_value_list(filename, lst):
    values = []
    data = sk_LR.load_features(filename)
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1, 10))
    # data = min_max_scaler.fit_transform(data)
    #data = data.dot(10)
    for row in data:
        sum = 0
        i = 0
        count = 0
        for n in row:
            if lst[i] != 0:
                sum += n
                count += 1
            i += 1
            if count == 0:
                pass
        values.append(sum / count)
    return values


if __name__ == '__main__':

    golden_score = post.read_score(macro.CORPUS_DIR + '/500_2.csv')
    lst = [0, 0, 0, 0, 0, 0, 0]
    best = []
    i = 0
    max = 0
    while i < 127:
        add(lst)
        data = get_value_list(macro.CORPUS_DIR + '/features_golden_new.txt', lst)
        sp = eval.spearman(data, golden_score)[0]
        if sp > max:
            max = sp
            best = lst
        # if sp > 0.3:
        #     dataset = {
        #         'cal_value': data,
        #         'goldern': golden_score
        #     }
        #     frame = DataFrame(dataset)
        #     sns.jointplot('goldern', 'cal_value', frame, kind='reg', stat_func=eval.spearmanr)
        #
        #     plt.xlim([1,10])
        #     plt.ylim([1,10])
        #     plt.savefig('%s/%s.png' %(macro.PICS_DIR,str(lst)))
        # print lst, sp
        i += 1
    print max,best