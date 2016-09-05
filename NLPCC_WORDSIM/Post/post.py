# encoding=UTF-8
"""
    @author: Zeco on 2016/7/5
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
import codecs

import string
from sklearn import preprocessing

from Proc.LR import sk_LR
from Com import macro
from Eval import eval
from Com import utils
import merge
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt


# 读取特征文件，放缩各个特征后计算一个相似度得分 lst:开关列表
# [0, 0, 0, 1, 0, 1, 0] 0.320099049922
def get_value_list(filename, lst):
    values = []
    data = sk_LR.load_features(filename)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1, 10))
    data = min_max_scaler.fit_transform(data)
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



# 从文件中读取相似度得分，默认读取一万条，文件不足一万条则全部读取，读取数量由num指定
def read_score(filename, num=10000):
    score = []
    i = 0
    infile = codecs.open(filename, 'r', 'utf-8')
    lines = infile.readlines()[1:]
    for line in lines:
        if i >= num:
            break
        i += 1
        words = line.strip().split('\t')
        if len(words) < 3:
            break
        score.append(string.atof(words[-1].strip()))

    infile.close()
    return score


def write_list_2_file(lst, file_name):
    outfile = codecs.open(file_name, 'w', 'utf-8')
    for element in lst:
        outfile.write(element)
        outfile.write('\r\n')
    outfile.close()


def extract():
    file_old = codecs.open(macro.RESULTS_DIR + '/merge_result.txt', 'r', 'utf-8')
    file_list = codecs.open(macro.CORPUS_DIR + '/NLPCC_Formal500.txt', 'r', 'utf-8')
    out_file = codecs.open(macro.CORPUS_DIR + '/merge_result_extract.txt', 'w', 'utf-8')
    lines_old = file_old.readlines()[1:]
    lines_list = file_list.readlines()[1:]

    for line_l in lines_list:
        words_l = line_l.strip().split('\t')
        id_l = words_l[0]
        for line_o in lines_old:
            words_o = line_o.strip().split('\t')
            id_o = words_o[0]
            if id_o == id_l:
                out_file.write(line_o)
    out_file.close()
    file_list.close()
    file_old.close()

def small(pred):
    results = []
    sum = 0
    count = 0
    for p in pred:
        results.append(p)
        if p!=10:
            sum+=p
            count+=1
    mean = sum*1.0/count
    for i in range(0,len(results)):
        if results[i]==10:
            results[i] = mean
    return results

def merge2max(s1, s2):
    result = []
    for ss1, ss2 in zip(s1, s2):
        result.append(max([ss1, ss2]))
    return result

if __name__ == '__main__':
    golden_score = read_score(macro.CORPUS_DIR + '/500_2.csv')
    # data = get_value_list(macro.CORPUS_DIR + '/test.txt')
    # i = 0
    #
    f_c = macro.RESULTS_DIR + '/evatestdata3_goldern500_cilin.txt'
    f_v = macro.RESULTS_DIR + '/fml_org_google_en_w2v_org.result'
    # f_m = macro.RESULTS_DIR + '/merge_result_new.txt'
    #
    # score_c = read_score(f_c)
    # data = get_value_list(macro.CORPUS_DIR + '/features_golden_new.txt')
    #
    # score_v = read_score(f_v)
    #
    # lst = [0, 0, 0, 1, 0, 1, 0]
    # data = get_value_list(macro.CORPUS_DIR + '/features_golden_new.txt',lst)
    # max = 0
    # # final_list = []
    # for mode in range(1, 13):
    #     score_m = merge.merge_2_list(f_v, f_c, mode)
    #     temp = eval.spearman(golden_score, score_m)[0]
    #     temp2 = eval.pearson(golden_score,score_m)[0]
    #     if temp > max:
    #         max = temp
    #         final_list = score_m
    #     print '合并：',macro.MODES[mode - 1]+'/fml_org_google_en_w2v_org.result\t',temp,'\t',temp2
        # print macro.MODES[mode-1] + ' vs cal_value: ', eval.spearman(score_m,data)[0]
    # outfile = codecs.open(macro.CORPUS_DIR+'/replace1andaverage.txt','w','utf-8')
    # for f in final_list:
    #     outfile.write(str(f)+'\r\n')

    # score_m = merge.merge_2_list(f_v,f_c,macro.REPLACE_1_AND_AVERAGE)
    # print eval.spearman(score_m,golden_score)



    # merge.merge(f_v, f_c, f_m)
    # score_m = read_score(f_m, 3644)
    #
    # print len(score_v)
    #
    # print 'Vector: ', eval.spearman(data, score_v)
    # print 'Cilin: ', eval.spearman(data, score_c)
    # print 'merge: ', eval.spearman(data, score_m)
    # score_m = read_score(macro.CORPUS_DIR+'/merge_result_extract.txt')
    # print golden_score
    # print score_m
    # print spearmanr(golden_score,score_m)
    # eval.spearman(golden_score,score_m)
    last_scores = []
    max_score = []
    #
    #
    for i in range(1, 6):
        last_scores.append(merge.merge_2_list(macro.RESULTS_DIR + '/lstm_w2v' + str(i) + '.txt',f_c,mode=macro.MAX))
    idl, w1l, w2l, score_goldern, headline = utils.read2wordlist([(macro.CORPUS_DIR, '500_2.csv')])
    temp = last_scores[0]
    for s in last_scores[1:]:
        max_score = merge2max(temp, s)
        temp = max_score
    print ('max_score: ', eval.spearman(max_score, score_goldern), eval.pearson(max_score, score_goldern))
    sss = small(max_score)
    print eval.spearman(sss,score_goldern)
    # dataset = {
    #     'pred': max_score,
    #     'goldern': score_goldern
    # }
    # frame = DataFrame(dataset)
    # sns.jointplot('goldern', 'pred', frame, kind='reg', stat_func=eval.spearman)
    #
    # plt.xlim([1, 10])
    # plt.ylim([1, 10])
    # plt.savefig('%s/%s.png' % (macro.PICS_DIR, ('cilin_w2v_trans_lstm_max')))
    # pass
