# encoding=UTF-8
"""
    @author: PP-window on 2016/5/25
    @email: ppsunrise99@gmail.com
    @step: 
    @function: 用预先训练好的词向量计算词的相似度
"""
from Com import utils, macro
from Eval import eval
from gensim.models import Word2Vec
import numpy as np
from sklearn import preprocessing


class WordSim:
    def __init__(self, f_tuple_list, w2v_model_file, result_fname):
        self.f_tuple_list = f_tuple_list
        self.word1_list, self.word2_list, self.manu_sim_list, self.headline = utils.read2wordlist(self.f_tuple_list)
        self.ofname = result_fname    # f_tuple_list存着dir,file的对儿
        self.w2v_model_file = w2v_model_file

    def wordvec_sim(self, write_flag=True):
        print 'load wordvec model:%s/%s' % (macro.DICT_DIR, self.w2v_model_file)
        w2v_model = Word2Vec.load_word2vec_format(r'%s/%s' % (macro.DICT_DIR, self.w2v_model_file), binary=True)  # C format
        auto_sim_list = []
        for w1, w2, manu_sim in zip(self.word1_list, self.word2_list, self.manu_sim_list):
            try:
                auto_sim = w2v_model.similarity(w1, w2)  # 将余弦相似度放到0-10得分
                if auto_sim <= 0:
                    auto_sim = 1.0
                else:
                    auto_sim = auto_sim*9+1
                # print '%-10s\t%-10s\t%-10s\t%-10s' % (w1, w2, manu_sim, auto_sim)
            except:
                auto_sim = 1                            # cos值的最小值
                print '%-10s\t%-10s\t%-10s\t%-10s' % (w1, w2, manu_sim, '______Not Found______')
            auto_sim_list.append(auto_sim)

        for w1, w2, manu_sim,auto_sim in zip(self.word1_list, self.word2_list, self.manu_sim_list, auto_sim_list):
            print '%-10s\t%-10s\t%-10s\t%-10s' % (w1, w2, manu_sim, auto_sim)

        if write_flag:
            print 'write result to file...'
            with open('%s/%s' % (macro.RESULTS_DIR, self.ofname), 'w') as fw:
                fw.write(self.headline.strip()+'\tauto_sim_score\n')
                for w1, w2, manu_sim, auto_sim in zip(self.word1_list, self.word2_list, self.manu_sim_list, auto_sim_list):
                    fw.write('%s\t%s\t%s\t%s\n' % (w1, w2, manu_sim, auto_sim))

        return self.word1_list, self.word2_list, self.manu_sim_list, auto_sim_list, self.headline

    # 调用评价函数
    def evalutaion(self):
        print 'call eval function...'
        eval.evaluate(macro.RESULTS_DIR, self.ofname, mode='spearman')

if __name__ == '__main__':
    pass

