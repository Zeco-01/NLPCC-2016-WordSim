# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""

from Com import macro
from Eval import eval
from Com import utils
import post
import merge

lst = [1] * 7

data = post.get_value_list(macro.CORPUS_DIR + '/features_golden_new.txt', lst)
max = 0
final_list = []
idl, w1l, w2l, score, headline = utils.read2wordlist([(macro.CORPUS_DIR, '500_2.csv')])
f_c = macro.RESULTS_DIR + '/evatestdata3_goldern500_cilin.txt'
f_v = macro.RESULTS_DIR + '/fml_org_bdnews_xieso.result'

for mode in range(1, 13):
    score_m = merge.merge_2_list(f_v, f_c, mode)
    sp = eval.spearman(data, score_m)[0]
    pe = eval.pearson(data, score_m)[0]
    temp = score_m
    print macro.MODES[mode - 1], '\t', eval.spearman(score, score_m)[0], '\t', eval.pearson(score, score_m)[
        0], '\t', sp, '\t', pe
    # idl_p, w1l_p, w2l_p, score_p, headline_p = utils.read2wordlist([(macro.RESULTS_DIR,'best_without_lstm.txt')])

    # pred = merge.merge_2_list(macro.RESULTS_DIR+'/fml_google_en_w2v.result',f_c,mode=macro.MAX)
    # print eval.spearman(pred,score),eval.pearson(pred,score)

    # merge.merge(macro.RESULTS_DIR+'/fml_google_en_w2v.result',f_c,macro.RESULTS_DIR+'/best_without_lstm.txt',macro.MAX)
