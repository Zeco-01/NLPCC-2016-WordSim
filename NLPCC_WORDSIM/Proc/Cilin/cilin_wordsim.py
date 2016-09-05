# encoding=UTF-8
"""
    @author: Administrator on 2016/6/23
    @email: ppsunrise99@gmail.com
    @step:
    @function: 利用同义词林计算相似度的代码在java中实现，这里只是算spearman指数
"""
from Com import macro
from Eval import eval

if __name__ == '__main__':
    # 同义词林计算dry run数据
    ofname = macro.NLPCC_DRY_FILE.split('.')[0]+'_cilin.txt'
    print 'spearman', eval.evaluate(macro.RESULTS_DIR, ofname, mode='spearman')
    print 'pearson', eval.evaluate(macro.RESULTS_DIR, ofname, mode='pearson')

    # 同义词林计算带有答案的formal run数据
    # ofname = macro.NLPCC_FML_GD_FILE.split('.')[0] + '_cilin.txt'
    # print 'spearman', eval.evaluate(macro.RESULTS_DIR, ofname, mode='spearman')
    # print 'pearson', eval.evaluate(macro.RESULTS_DIR, ofname, mode='pearson')

