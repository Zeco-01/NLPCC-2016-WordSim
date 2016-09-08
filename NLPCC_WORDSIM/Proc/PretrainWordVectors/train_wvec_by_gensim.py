# encoding=UTF-8
"""
    @author: pp on 2016/6/23
    @email: ppsunrise99@gmail.com
    @step:
    @function: 用gensim训练词向量
"""
from Com import macro, utils
from Eval import eval
from gensim.models.word2vec import Word2Vec
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas import DataFrame
from translate import Translator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MyWordVec:
    def __init__(self, f_tuple_list, seg_docs_dir_list, mode='tag'):
        self.mode = mode  # tag:有正确答案的，格式为(ID,Word1,Word2,Score,Prediction)；no_tag:没有正确答案的，格式为(ID,Word1,Word2,Score)
        self.f_tuple_list = f_tuple_list  # 评价的词对文件，[(dir_1,file_1),...,(dir_n,file_n)]
        self.seg_docs_dir_list = seg_docs_dir_list  # 分好词的语料文件，[(dir_1,file_1),...,(dir_n,file_n),dir_m]

    # 训练语料得到w2v模型
    def train_model(self, save_model, fvocab='', dim=300):
        # 抽取语料中的句子放入list，每个元素是分好词的句子
        sentences = []
        for seg_docs_dir in self.seg_docs_dir_list:
            if type(seg_docs_dir) == tuple:  # 如果是一个文件(dir,file)，则将指定文件的内容作为语境语料
                sens = utils.atxt2sens(seg_docs_dir[0], seg_docs_dir[1])
            else:  # 如果是目录dir:默认只挑选出评测词表对应的语境语料进行训练；如果指定词表文件，则按照词表筛选语境
                sens = utils.f_tuple_list2sens(self.f_tuple_list, seg_docs_dir, fvocab, self.mode)
            sentences.extend(sens)
        # gensim训练w2v model
        model = Word2Vec(sentences, sg=1, size=dim, window=10, negative=0, hs=1, sample=1e-4, workers=20, min_count=5)
        # 保存model
        model.save_word2vec_format('%s/%s' % (macro.MODELS_DIR, save_model), binary=True)  # 保存模型
        return model

    # 加载预先训练好的大规模语料的词表,计算相似度,评价性能
    def calculate_sim(self, load_model, ofname, write_flag=True):
        # 加载指定w2v model
        w2v_model = Word2Vec.load_word2vec_format(r'%s/%s' % (macro.MODELS_DIR, load_model), binary=True)  # C format
        # 读入评测词对语料
        id_list, word1_list, word2_list, manu_sim_list, headline = utils.read2wordlist(self.f_tuple_list, mode='tag')
        # 新的题头
        new_headline = headline.strip() + '\tPrediction\n'
        # 计算相似度
        auto_sim_list = []
        for w1, w2, manu_sim in zip(word1_list, word2_list, manu_sim_list):
            try:
                auto_sim = w2v_model.similarity(w1, w2)  # 向量余弦相似度[-1,1]
                auto_sim = utils.convert_sim(auto_sim)  # 将余弦相似度放到1-10得分
                print '%-10s\t%-10s\t%-10s\t%-10s' % (w1, w2, manu_sim, auto_sim)
            except:
                auto_sim = 1  # 未登录词，为了区分1.0，赋值为１
                print '%-10s\t%-10s\t%-10s\t%-10s' % (w1, w2, manu_sim, '______Not Found______')
            auto_sim_list.append(auto_sim)

        # 相似度计算的结果是否写入文件
        if write_flag:
            print 'write result to file...'
            with open('%s/%s' % (macro.RESULTS_DIR, ofname), 'w') as fw:
                fw.write(new_headline)
                for w1, w2, manu_sim, auto_sim in zip(word1_list, word2_list, manu_sim_list, auto_sim_list):
                    fw.write('%s\t%s\t%s\t%s\n' % (w1.encode('utf-8'), w2.encode('utf-8'), manu_sim, auto_sim))

        # 评价结果
        r = eval.spearmanr(manu_sim_list, auto_sim_list)
        p = eval.pearsonr(manu_sim_list, auto_sim_list)
        print '!!!spearman=%s; pearson=%s' % (r, p)

        # 可视化结果
        data = {'ID': id_list,
                'Word1': word1_list,
                'Word2': word2_list,
                'Score': manu_sim_list,
                'Prediction': auto_sim_list}

        frame = DataFrame(data)
        sns.jointplot("Score", "Prediction", frame, kind='reg', stat_func=eval.spearmanr)
        plt.savefig('%s/%s.jpg' % (macro.PICS_DIR, ofname))

        return word1_list, word2_list, manu_sim_list, auto_sim_list, new_headline

    # 加载预先训练好的大规模语料的词表,计算相似度
    def calculate_sim_without_tag(self, load_model, ofname, write_flag=True):
        # 加载指定w2v model
        w2v_model = Word2Vec.load_word2vec_format(r'%s/%s' % (macro.MODELS_DIR, load_model), binary=True)  # C format
        # 读入评测词对语料
        id_list, word1_list, word2_list, headline = utils.read2wordlist(self.f_tuple_list, mode='no_tag')
        # 新的题头
        new_headline = headline.strip() + '\tPrediction\n'
        # 计算相似度
        auto_sim_list = []
        for w1, w2 in zip(word1_list, word2_list):
            try:
                auto_sim = w2v_model.similarity(w1, w2)  # 向量余弦相似度[-1,1]
                auto_sim = utils.convert_sim(auto_sim)  # 将余弦相似度放到1-10得分
                print '%-10s\t%-10s\t%-10s' % (w1, w2, auto_sim)
            except:
                auto_sim = 1  # 未登录词，为了区分1.0，赋值为１
                print '%-10s\t%-10s\t%-10s' % (w1, w2, '______Not Found______')
            auto_sim_list.append(auto_sim)

        # 相似度计算的结果是否写入文件
        if write_flag:
            print 'write result to file...'
            with open('%s/%s' % (macro.RESULTS_DIR, ofname), 'w') as fw:
                fw.write(new_headline)
                for w1, w2, auto_sim in zip(word1_list, word2_list, auto_sim_list):
                    fw.write('%s\t%s\t%s\n' % (w1, w2, auto_sim))

        return word1_list, word2_list, auto_sim_list, new_headline

    # 训练同义词林扩展的词表对应的句子＋large corpus对应的句子，而且要根据评价指标迭代若干次选最优
    def train_ext_vocab_choose_best(self, save_model, result_fname, last_val):
        # 获取评价词对
        id_list, word1_list, word2_list, manu_sim_list, headline = utils.read2wordlist(self.f_tuple_list, mode='tag')

        # 获取语料
        sentences = []
        for seg_docs_dir in self.seg_docs_dir_list:
            if type(seg_docs_dir) == tuple:
                sens = utils.atxt2sens(seg_docs_dir[0], seg_docs_dir[1])
            else:
                sens = utils.txts2sens(seg_docs_dir)
            sentences.extend(sens)

        # 得到模型方式：load之前的模型 OR 训练词向量模型
        if last_val == -2:
            print 'load previous model....'
            model = Word2Vec.load_word2vec_format(r'%s/%s' % (macro.MODELS_DIR, save_model), binary=True)
        else:
            model = Word2Vec(sentences, sg=1, size=300, window=10, negative=0, hs=1, sample=1e-4, workers=8,
                             min_count=5)

        # 评价相似度
        auto_sim_list = []
        for w1, w2, manu_sim in zip(word1_list, word2_list, manu_sim_list):
            try:
                auto_sim = model.similarity(w1, w2)  # 将余弦相似度放到1-10得分
                auto_sim = utils.convert_sim(auto_sim)
                # print '%-10s\t%-10s\t%-10s\t%-10s' % (w1, w2, manu_sim, auto_sim)
            except:
                auto_sim = 1  # 为了区分没有找到的情况，用１代替1.0
                print '%-10s\t%-10s\t%-10s\t%-10s' % (w1, w2, manu_sim, '______Not Found______')
            auto_sim_list.append(auto_sim)

        # 保留val大的模型
        val = eval.spearman(manu_sim_list, auto_sim_list)
        if val > last_val:
            model.save_word2vec_format('%s/%s' % (macro.MODELS_DIR, save_model), binary=True)  # 保存模型
            print 'write result to file...'
            with open('%s/%s' % (macro.RESULTS_DIR, result_fname), 'w') as fw:
                fw.write(headline.strip() + '\tPrediction\n')
                for w1, w2, manu_sim, auto_sim in zip(word1_list, word2_list, manu_sim_list, auto_sim_list):
                    fw.write('%s\t%s\t%s\t%s\n' % (w1.encode('utf-8'), w2.encode('utf-8'), manu_sim, auto_sim))
        else:
            print ':::::::current val=', val
        return val

    # 迭代，找到最佳
    def iteration(self, iter, save_model, ofname, last_val=-1):
        start = time.clock()
        val = self.train_ext_vocab_choose_best(save_model, ofname, last_val)
        end = time.clock()
        print 'total time = %ss' % (end - start)
        print 'iter=0;\tval=%s' % val

        for i in range(iter)[1:]:
            start = time.clock()
            val = self.train_ext_vocab_choose_best(save_model, ofname, val)
            end = time.clock()
            print 'total time = %ss' % (end - start)
            print 'iter=%s;\tval=%s' % (i, val)


# ===============================Dry Run==========================================
# 1. NLPCC样例数据+数据堂语料
def dry_datatang():
    print '======================>NLPCC样例数据+大规模互联网语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评测词对语料
    seg_docs_dir_list = [(macro.BIG_CORPUS_SEG_DIR, macro.DATATANG_SEG_FILE)]  # 分好词的语料文档或者文档所在的所有目录
    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_DATATANG_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_DATATANG_RESULT, write_flag=write_flag)


# 2. NLPCC样例数据_维基百科语料
def dry_wiki():
    print '======================>2. NLPCC样例数据_维基百科语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评测词对语料
    seg_docs_dir_list = [(macro.BIG_CORPUS_SEG_DIR, macro.WIKI_SEG_FILE)]  # 分好词的语料文档或者文档所在的所有目录
    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_WIKI_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_WIKI_RESULT, write_flag=write_flag)


# 3.1 NLPCC样例数据_原始词表_爬取的百度新闻语料
def dry_org_bdnews():
    print '======================>3.1 NLPCC样例数据_原始词表_爬取的百度新闻语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_ORG_BDNEWS_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.DRY_ORG_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_ORG_BDNEWS_RESULT, write_flag=write_flag)


# 3.2 NLPCC样例数据_cilin扩展词表_爬取的百度新闻语料
def dry_ext_bdnews():
    print '======================>3.2 NLPCC样例数据_cilin扩展词表_爬取的百度新闻语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_EXT_BDNEWS_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.DRY_EXT_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_EXT_BDNEWS_RESULT, write_flag=write_flag)


# 4.1 NLPCC样例数据_原始词表_爬取的www.xieso.net造句语料
def dry_org_xieso():
    print '======================>4.1 NLPCC样例数据_原始词表_爬取的www.xieso.net造句语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.XIESO_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_ORG_XIESO_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.DRY_ORG_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_ORG_XIESO_RESULT, write_flag=write_flag)


# 4.2 NLPCC样例数据_扩展词表_爬取的www.xieso.net造句语料
def dry_ext_xieso():
    print '======================>4.1 NLPCC样例数据_原始词表_爬取的www.xieso.net造句语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.XIESO_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_EXT_XIESO_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.DRY_EXT_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_EXT_XIESO_RESULT, write_flag=write_flag)


# 5.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料
def dry_org_bdnews_xieso():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_ORG_BDNEWS_XIESO_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.DRY_ORG_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_ORG_BDNEWS_XIESO_RESULT, write_flag=write_flag)


# 5.2 NLPCC样例数据_扩展词表_百度新闻语料_爬取的www.xieso.net造句语料
def dry_ext_bdnews_xieso():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_EXT_BDNEWS_XIESO_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.DRY_EXT_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_EXT_BDNEWS_XIESO_RESULT, write_flag=write_flag)


# 6.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_数据堂语料
def dry_org_bdnews_xieso_datatang():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.DATATANG_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_ORG_BDNEWS_XIESO_DATATANG_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.DRY_ORG_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_ORG_BDNEWS_XIESO_DATATANG_RESULT,
                           write_flag=write_flag)


# 6.2 NLPCC样例数据_扩展词表_百度新闻语料_爬取的www.xieso.net造句语料_数据堂语料
def dry_ext_bdnews_xieso_datatang():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.DATATANG_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_EXT_BDNEWS_XIESO_DATATANG_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.DRY_EXT_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_EXT_BDNEWS_XIESO_DATATANG_RESULT,
                           write_flag=write_flag)


# 7.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
def dry_org_bdnews_xieso_wiki():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.WIKI_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_ORG_BDNEWS_XIESO_WIKI_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.DRY_ORG_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_ORG_BDNEWS_XIESO_WIKI_RESULT, write_flag=write_flag)


# 7.2 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
def dry_ext_bdnews_xieso_wiki():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.WIKI_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_EXT_BDNEWS_XIESO_WIKI_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = True
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.DRY_EXT_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.DRY_EXT_BDNEWS_XIESO_WIKI_RESULT, write_flag=write_flag)


# 8.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
def dry_org_bdnews_xieso_datatang_wiki():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR,
                         macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.DATATANG_SEG_FILE),
                         (macro.BIG_CORPUS_SEG_DIR, macro.WIKI_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_ORG_BDNEWS_XIESO_DATATANG_WIKI_W2V_MODEL
    fvocab = macro.DRY_ORG_VOCAB_DICT
    ofname = macro.DRY_ORG_BDNEWS_XIESO_DATATANG_WIKI_RESULT
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=fvocab)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=ofname, write_flag=write_flag)


# 8.2 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
def dry_ext_bdnews_xieso_datatang_wiki():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR,
                         macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.DATATANG_SEG_FILE),
                         (macro.BIG_CORPUS_SEG_DIR, macro.WIKI_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_EXT_BDNEWS_XIESO_DATATANG_WIKI_W2V_MODEL
    fvocab = macro.DRY_EXT_VOCAB_DICT
    ofname = macro.DRY_EXT_BDNEWS_XIESO_DATATANG_WIKI_RESULT
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=fvocab)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=ofname, write_flag=write_flag)


# =================================Formal Run========================================
# -1. NLPCC正式数据+数据堂语料
def formal_datatang():
    print '======================>NLPCC样例数据+大规模互联网语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评测词对语料
    seg_docs_dir_list = [(macro.BIG_CORPUS_SEG_DIR, macro.DATATANG_SEG_FILE)]  # 分好词的语料文档或者文档所在的所有目录
    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_DATATANG_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = True
    else:
        mw2v_obj.train_model(save_model=model_name)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_DATATANG_RESULT, write_flag=write_flag)


# -2. NLPCC样例数据_维基百科语料
def formal_wiki():
    print '======================>2. NLPCC样例数据_维基百科语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评测词对语料
    seg_docs_dir_list = [(macro.BIG_CORPUS_SEG_DIR, macro.WIKI_SEG_FILE)]  # 分好词的语料文档或者文档所在的所有目录
    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.DRY_WIKI_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = True
    else:
        mw2v_obj.train_model(save_model=model_name)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_WIKI_RESULT, write_flag=write_flag)


# -3.1 NLPCC样例数据_原始词表_爬取的百度新闻语料
def formal_org_bdnews():
    print '======================>3.1 NLPCC样例数据_原始词表_爬取的百度新闻语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_ORG_BDNEWS_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.FML_ORG_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_ORG_BDNEWS_RESULT, write_flag=write_flag)


# -3.2 NLPCC样例数据_cilin扩展词表_爬取的百度新闻语料
def formal_ext_bdnews():
    print '======================>3.2 NLPCC样例数据_cilin扩展词表_爬取的百度新闻语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_EXT_BDNEWS_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.FML_EXT_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_EXT_BDNEWS_RESULT, write_flag=write_flag)


# -4.1 NLPCC样例数据_原始词表_爬取的www.xieso.net造句语料
def formal_org_xieso():
    print '======================>4.1 NLPCC样例数据_原始词表_爬取的www.xieso.net造句语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.XIESO_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_ORG_XIESO_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.FML_ORG_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_ORG_XIESO_RESULT, write_flag=write_flag)


# -4.2 NLPCC样例数据_扩展词表_爬取的www.xieso.net造句语料
def formal_ext_xieso():
    print '======================>4.1 NLPCC样例数据_原始词表_爬取的www.xieso.net造句语料'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.XIESO_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_EXT_XIESO_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.FML_EXT_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_EXT_XIESO_RESULT, write_flag=write_flag)


# -5.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料
def formal_org_bdnews_xieso():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_ORG_BDNEWS_XIESO_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.FML_ORG_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_ORG_BDNEWS_XIESO_RESULT, write_flag=write_flag)


# -5.2 NLPCC样例数据_扩展词表_百度新闻语料_爬取的www.xieso.net造句语料
def formal_ext_bdnews_xieso():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_EXT_BDNEWS_XIESO_W2V_MODEL
    ofname = macro.FML_EXT_BDNEWS_XIESO_RESULT
    dim = 300
    fvocab = macro.FML_EXT_VOCAB_DICT
    write_flag = True

    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=fvocab, dim=dim)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=ofname, write_flag=write_flag)


# -6.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_数据堂语料
def formal_org_bdnews_xieso_datatang():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.DATATANG_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_ORG_BDNEWS_XIESO_DATATANG_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.FML_ORG_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_ORG_BDNEWS_XIESO_DATATANG_RESULT,
                           write_flag=write_flag)


# -6.2 NLPCC样例数据_扩展词表_百度新闻语料_爬取的www.xieso.net造句语料_数据堂语料
def formal_ext_bdnews_xieso_datatang():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.DATATANG_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_EXT_BDNEWS_XIESO_DATATANG_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.FML_EXT_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_EXT_BDNEWS_XIESO_DATATANG_RESULT,
                           write_flag=write_flag)


# -7.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
def formal_org_bdnews_xieso_wiki():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.WIKI_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_ORG_BDNEWS_XIESO_WIKI_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.FML_ORG_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_ORG_BDNEWS_XIESO_WIKI_RESULT, write_flag=write_flag)


# -7.2 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
def formal_ext_bdnews_xieso_wiki():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.WIKI_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_EXT_BDNEWS_XIESO_WIKI_W2V_MODEL
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=macro.FML_EXT_VOCAB_DICT)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=macro.FML_EXT_BDNEWS_XIESO_WIKI_RESULT, write_flag=write_flag)


# -8.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
def formal_org_bdnews_xieso_datatang_wiki():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR,
                         macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.DATATANG_SEG_FILE),
                         (macro.BIG_CORPUS_SEG_DIR, macro.WIKI_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_ORG_BDNEWS_XIESO_DATATANG_WIKI_W2V_MODEL
    fvocab = macro.FML_ORG_VOCAB_DICT
    ofname = macro.FML_ORG_BDNEWS_XIESO_DATATANG_WIKI_RESULT
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=fvocab)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=ofname, write_flag=write_flag)


# -8.2 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
def formal_ext_bdnews_xieso_datatang_wiki():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR,
                         macro.XIESO_DOCS_SEG_DIR,
                         (macro.BIG_CORPUS_SEG_DIR, macro.DATATANG_SEG_FILE),
                         (macro.BIG_CORPUS_SEG_DIR, macro.WIKI_SEG_FILE)]  # 分词文档所在的所有目录

    # 对象实例化(评测词对语料，分好词的语境语料，有无标签)
    mw2v_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode='tag')

    model_name = macro.FML_EXT_BDNEWS_XIESO_DATATANG_WIKI_W2V_MODEL
    fvocab = macro.FML_EXT_VOCAB_DICT
    ofname = macro.FML_EXT_BDNEWS_XIESO_DATATANG_WIKI_RESULT
    write_flag = True
    # model不存在则训练词向量模型
    if os.path.exists('%s/%s' % (macro.MODELS_DIR, model_name)):
        write_flag = False
    else:
        mw2v_obj.train_model(save_model=model_name, fvocab=fvocab)

    # 计算相似度，并默认将结果写入文件
    mw2v_obj.calculate_sim(load_model=model_name, ofname=ofname, write_flag=write_flag)


# 根据dry run数据迭代,得到n次迭代内最好的model
def dry_run_best():
    mode = 'NLPCC_DRY'
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]  # 评价的多个文件
    seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR, macro.XIESO_DOCS_SEG_DIR,
                         (macro.DICT_DIR, macro.SML_CPS_SEG_FILE)]  # 分词文档所在的所有目录
    # seg_docs_dir_list = [macro.BDNEWS_DOCS_SEG_DIR]  # 分词文档所在的所有目录

    # 参数　f_tuple_list, w2v_model_file, seg_docs_dir_list, mode = 'NLPCC_DRY'
    train_obj = MyWordVec(f_tuple_list, seg_docs_dir_list, mode)
    # train_obj.iteration(iter=10, save_model=macro.DRY_EXT_MIX_BST_W2V_MODEL,ofname=macro.DRY_EXT_MIX_BST_RESULT, last_val=-1) # last_val=-1则从头训练
    # load已有模型
    train_obj.iteration(iter=10, save_model=macro.DRY_EXT_MIX_BST_W2V_MODEL, ofname=macro.DRY_EXT_MIX_BST_RESULT,
                        last_val=-1)


# 评测提交比赛数据
def test_formal_run():
    # 读入已知词对，如果新评测词对在已知中，则直接取出
    gordern_word1_list, gordern_word2_list, manu_sim_list, headline1 = utils.read2wordlist(
        [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)], 'NLPCC_DRY')
    mode = 'NLPCC_FML'
    id_list, word1_list, word2_list, headline2 = utils.read2wordlist([(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)], mode)
    model = Word2Vec.load_word2vec_format(r'%s/%s' % (macro.DICT_DIR, macro.DRY_EXT_MIX_BST_W2V_MODEL), binary=True)
    auto_sim_list = []
    for id, w1, w2 in zip(id_list, word1_list, word2_list):
        if w1 in gordern_word1_list and w2 == gordern_word2_list[gordern_word1_list.index(w1)]:
            auto_sim = manu_sim_list[gordern_word1_list.index(w1)]
            print 'found it in dry run data:::(%s\t%s\t%s)' % (w1, w2, auto_sim)
        elif w2 in gordern_word1_list and w1 == gordern_word2_list[gordern_word1_list.index(w2)]:
            auto_sim = manu_sim_list[gordern_word1_list.index(w1)]
            print 'found it in dry run data:::(%s\t%s\t%s)' % (w1, w2, auto_sim)
        else:
            try:
                auto_sim = model.similarity(w1, w2)  # 将余弦相似度放到0-10得分
                if auto_sim <= 0:
                    auto_sim = 1.0
                else:
                    auto_sim = auto_sim * 9 + 1
                # auto_sim = 0.5*(auto_sim+1)*10
                print '%-10s\t%-10s\t%-10s\t%-10s' % (id, w1, w2, auto_sim)
            except:
                auto_sim = 1
                print '%-10s\t%-10s\t%-10s\t%-10s' % (id, w1, w2, '______Not Found______')
        auto_sim_list.append(auto_sim)

    # 写入文件
    fw = open('%s/%s' % (macro.RESULTS_DIR, macro.FML_EXT_MIX_BST_RESULT), 'w')
    fw.write(headline2)
    for id, w1, w2, auto_sim in zip(id_list, word1_list, word2_list, auto_sim_list):
        fw.write('%s\t%s\t%s\t%s\n' % (id.encode('utf-8'), w1.encode('utf-8'), w2.encode('utf-8'), auto_sim))
    print 'test_formal_run:::finished!'
    return


# 测试已知答案的500条，最终评测结果
def final_test_formal_run():
    # 读入已知词对，如果新评测词对在已知中，则直接取出
    id_list1, gordern_word1_list, gordern_word2_list, manu_sim_list, headline1 = utils.read2wordlist(
        [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)], 'tag')
    # 模式是带有答案的
    id_list, word1_list, word2_list, manu_sim_list2, headline2 = utils.read2wordlist(
        [(macro.CORPUS_DIR, macro.NLPCC_FML_GD_FILE)], 'tag')
    model = Word2Vec.load_word2vec_format(r'%s/%s' % (macro.MODELS_DIR, macro.DRY_EXT_MIX_BST_W2V_MODEL), binary=True)
    auto_sim_list = []
    for id, w1, w2 in zip(id_list, word1_list, word2_list):
        if w1 in gordern_word1_list and w2 == gordern_word2_list[gordern_word1_list.index(w1)]:
            auto_sim = manu_sim_list[gordern_word1_list.index(w1)]
            print 'found it in dry run data:::(%s\t%s\t%s)' % (w1, w2, auto_sim)
        elif w2 in gordern_word1_list and w1 == gordern_word2_list[gordern_word1_list.index(w2)]:
            auto_sim = manu_sim_list[gordern_word1_list.index(w1)]
            print 'found it in dry run data:::(%s\t%s\t%s)' % (w1, w2, auto_sim)
        else:
            try:
                auto_sim = model.similarity(w1, w2)  # 将余弦相似度放到0-10得分
                if auto_sim <= 0:
                    auto_sim = 1.0
                else:
                    auto_sim = auto_sim * 9 + 1
                # auto_sim = 0.5*(auto_sim+1)*10
                print '%-10s\t%-10s\t%-10s\t%-10s' % (id, w1, w2, auto_sim)
            except:
                auto_sim = 1
                print '%-10s\t%-10s\t%-10s\t%-10s' % (id, w1, w2, '______Not Found______')
        auto_sim_list.append(auto_sim)

    print eval.spearman(manu_sim_list2, auto_sim_list)
    # 写入文件
    fw = open('%s/%s' % (macro.RESULTS_DIR, macro.FNL_FML_EXT_MIX_BST_RESULT), 'w')
    fw.write(headline2)
    for id, w1, w2, auto_sim in zip(id_list, word1_list, word2_list, auto_sim_list):
        fw.write('%s\t%s\t%s\t%s\n' % (id.encode('utf-8'), w1.encode('utf-8'), w2.encode('utf-8'), auto_sim))
    print 'test_formal_run:::finished!'
    return


if __name__ == '__main__':
    # =============================Dry Run=====================================
    # 1. NLPCC样例数据_数据堂语料
    # dry_datatang()
    # 2. NLPCC样例数据_维基百科语料
    # dry_wiki()
    # 3.1 NLPCC样例数据+爬取的百度新闻语料
    # dry_org_bdnews()
    # 3.2 NLPCC样例数据_爬取的百度新闻语料_cilin扩展词表
    # 4.1 NLPCC样例数据_原始词表_爬取的www.xieso.net造句语料
    # dry_org_xieso()
    # 4.2 NLPCC样例数据_扩展词表_爬取的www.xieso.net造句语料
    # dry_ext_xieso()
    # 5.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料
    # dry_org_bdnews_xieso()
    # 5.2 NLPCC样例数据_扩展词表_百度新闻语料_爬取的www.xieso.net造句语料
    # dry_ext_bdnews_xieso()
    # 6.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_数据堂语料
    # dry_org_bdnews_xieso_datatang()
    # 6.2 NLPCC样例数据_扩展词表_百度新闻语料_爬取的www.xieso.net造句语料_数据堂语料
    # dry_ext_bdnews_xieso_datatang()
    # 7.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
    # dry_org_bdnews_xieso_wiki()
    # 7.2 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
    # dry_ext_bdnews_xieso_wiki()
    # 8.1 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
    # dry_org_bdnews_xieso_datatang_wiki()
    # 8.2 NLPCC样例数据_原始词表_百度新闻语料_爬取的www.xieso.net造句语料_wiki语料
    # dry_ext_bdnews_xieso_datatang_wiki()
    # ==============================Formal Run=======================================
    # formal_datatang()     # 这两个语料比较大，就直接用dry run训练的模型
    # formal_wiki()
    # formal_org_bdnews()
    # formal_ext_bdnews()
    # formal_org_xieso()
    # formal_ext_xieso()
    # formal_org_bdnews_xieso()
    formal_ext_bdnews_xieso()
    # formal_org_bdnews_xieso_datatang()
    # formal_ext_bdnews_xieso_datatang()
    # formal_org_bdnews_xieso_wiki()
    # formal_ext_bdnews_xieso_wiki()
    # formal_org_bdnews_xieso_datatang_wiki()
    # formal_ext_bdnews_xieso_datatang_wiki()
