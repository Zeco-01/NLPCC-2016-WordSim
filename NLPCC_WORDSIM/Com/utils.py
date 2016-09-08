# encoding=UTF-8
"""
    @author: Administrator on 2016/6/23
    @email: ppsunrise99@gmail.com
    @step:
    @function: 存放一些公用函数
"""
from Com import macro
from bs4 import BeautifulSoup
import re
import codecs
import math
import os
import numpy as np
import word_pair



# 读多个文件，提取word1_list, word2_list, manu_list
def read2wordlist(f_tuple_list, mode='tag'):
    headline = ''
    id_list, word1_list, word2_list, manu_sim_list = [], [], [], []
    lines = []
    for f_tuple in f_tuple_list:
        with open('%s/%s' % (f_tuple[0], f_tuple[1]), 'r') as fr:
            headline = fr.readline()  # 过滤第一行注释
            lines.extend(fr.readlines())
    if 'tag' == mode:
        # 带标记的数据
        for line in lines:
            id, word1, word2, manu_sim = line.decode('utf-8').strip().split('\t')
            id_list.append(id)
            word1_list.append(word1)
            word2_list.append(word2)
            manu_sim_list.append(np.float(manu_sim))
        return id_list, word1_list, word2_list, manu_sim_list, headline
    elif 'no_tag' == mode:
        # 带标记的数据
        for line in lines:
            id, word1, word2 = line.decode('utf-8').strip().split('\t')
            id_list.append(id)
            word1_list.append(word1)
            word2_list.append(word2)
        return id_list, word1_list, word2_list, headline


# 将同义词林读入二维列表
def read_cilin2list():
    fr = open('%s/%s' % (macro.DICT_DIR, macro.CILIN_DICT), 'r')
    cilin_list = []
    for line in fr.readlines():
        try:
            g = line.strip().split()
            if len(g) > 2:  # 至少除了这个词还有别的近义词
                # 中文必须要decode之再计算长度
                sim_set = [word.decode('utf-8') for word in g[1:] if
                           len(word.decode('utf-8')) > 1 and len(word.decode('utf-8')) < 5]
                cilin_list.append(sim_set)
        except:
            print 'err:::', line
    fr.close()
    # 只筛选词长在[1,4]的近义词
    return cilin_list


# 读入一个文本文件，存放进二维list的sens对象里
def atxt2sens(fdir, fname):
    sentences = []
    with open('%s/%s' % (fdir, fname)) as fr:
        for line in fr.readlines():
            sentences.append(line.decode('utf-8').strip().split())
    return sentences


# 读入一个目录下的所有文本文件，存放进二维list的sens对象里　
def txts2sens(fdir):
    sentences = []
    for fname in os.listdir(fdir):
        with open('%s/%s' % (fdir, fname)) as fr:
            for line in fr.readlines():
                sentences.append(line.decode('utf-8').strip().split())
    return sentences


# 输入文件列表，得到句子二维list
def f_tuple_list2sens(f_tuple_list, fdir, fvocab, mode='tag'):  # fdir是分好词的语境语料所在的目录，f_tuple_list是评测词对语料
    # 初始化
    word1_list, word2_list = [], []
    # 读入评测词对文件
    if 'tag' == mode:
        id_list, word1_list, word2_list, manu_sim_list, headline = read2wordlist(f_tuple_list, mode)
    elif 'no_tag' == mode:
        id_list, word1_list, word2_list, headline = read2wordlist(f_tuple_list, mode)
    # 获取词汇表:默认是根据评测词对语料构建，也可以通过vocab文件指定词表
    if fvocab == '':
        word_list = word1_list + word2_list
        vocab_list = list(sorted(set(word_list)))
    else:
        with open('%s/%s' % (macro.DICT_DIR, fvocab), 'r') as fr:
            vocab_list = [line.strip().decode('utf-8') for line in fr.readlines()]
    # 根据词汇表获取相应的句子
    sentences = []
    for fname in vocab_list:
        try:
            print 'FILE:::', fname, r'%s/%s.txt' % (fdir, fname)
            with open(r'%s/%s.txt' % (fdir, fname), 'r') as fr:
                for line in fr.readlines():
                    sentences.append(line.strip().split())
        except:
            pass
            print 'FILE_OPEN_ERR:::', fname, r'%s/%s.txt' % (fdir, fname)
    return sentences


# 使用不同的方案将计算出的sim值放缩到1-10得分
def convert_sim(auto_sim, mode=0):
    if 0 == mode:
        auto_sim = 4.5*auto_sim+5.5
    elif 1 == mode:
        if auto_sim <= 0:
            auto_sim = 1.0
        else:
            auto_sim = auto_sim * 9 + 1
    elif 2 == mode:  # 反双曲正切函数，直接把[-1,1]放缩
        auto_sim = math.atanh(auto_sim)
        if auto_sim < 1:
            auto_sim = 1
        elif auto_sim > 10:
            auto_sim = 10
    elif 3 == mode:
        auto_sim = 0.5*auto_sim*auto_sim+4.5*auto_sim+5
    return auto_sim


# 读取词对列表
def read_word_list(word_list_file_name):
    infile = codecs.open(word_list_file_name, 'r', 'utf-8')

    lines = infile.readlines()
    lines.remove(lines[0])
    word_pairs = []
    for line in lines:
        words = line.strip().split('\t')

        id = words[0]
        word1 = words[1]
        word2 = words[2]
        wp = word_pair.WordPair(id, word1, word2)
        word_pairs.append(wp)
    infile.close()
    return word_pairs

# 去除无用的标签
def remove_useless_tags(html):
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'img', 'a', 'head', 'li', 'style']):
        tag.extract()
    return soup.get_text()

def find_token(cut_list, char):
    if char in cut_list:
        return True
    else:
        return False


def join(l):
    text = ''
    for i in l:
        text += i
    return text


def cut(cut_list, text):
    l = []
    sentences = []
    for char in text:
        if find_token(cut_list, char):
            sentence = join(l)
            pattern = re.compile(r'\s+')
            if not pattern.match(sentence):  # and len(sentence) > 4:
                l.append(char)
                sentences.append(join(l))
                l = []
            else:
                l = []
        else:
            l.append(char)
    return sentences


if __name__ == '__main__':
    txts2sens(macro.BDNEWS_DOCS_SEG_DIR)
