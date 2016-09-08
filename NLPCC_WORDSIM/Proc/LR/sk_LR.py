# encoding=UTF-8
"""
    @author: Zeco on 2016/7/4
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from __future__ import division
import string
import requests
from bs4 import BeautifulSoup
import operator
import urllib
import math
import codecs
from Com import macro
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import numpy



# 是否有子串关系，是返回1，否返回0
def is_sub(word1, word2):
    if word1.find(word2) != -1 or word2.find(word1) != -1:
        return 1
    return 0


# 计算共享字数
def get_shared_cha_num(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    count = 0
    long_word = ''
    short_word = ''
    if len1 > len2:
        long_word = word1
        short_word = word2
    else:
        long_word = word2
        short_word = word1
    for char in short_word:
        if long_word.find(char) != -1:
            count += 1
    return count


# 从html中提取搜索结果数量 百度返回不超过一亿
def get_num_from_html(html):
    bs = BeautifulSoup(html, 'html.parser')
    divs = bs.find_all('div', class_='nums')
    t = ''
    if len(divs) > 0:
        div = divs[0]
        t = div.text
    else:
        return -1

    num = filter(operator.methodcaller('isdigit'), t)
    num = string.atof(num)

    return num


# def get_pattern_sim(word1,word2):
#     same_count = 0
#     sim_count = 0
#     if word1 == word2:
#         same_count = len(word1)
#         sim_count = 0
#     else:
#         for c in word1:


# 计算web-jaccard
def web_jaccard(p, q, pq):
    return pq / (p + q - pq)


# 计算web-overlap
def web_overlap(p, q, pq):
    return pq / (min([p, q]))


# 计算web-dice
def web_dice(p, q, pq):
    return (2 * pq) / (p + q)


# 计算web-pmi
def web_pmi(p, q, pq, N):
    print p, q, pq, N
    if pq < 5:
        return 0
    return math.log((N * pq) / (p * q), 2) / (math.log(N, 2))


# 计算四个根据结果数量计算的特征
def get_four_features(word1, word2):
    features = []  # web-jaccard, web-overlap, web-dice, web-pmi
    print word1 + ':::::::;' + word2
    headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'Accept-Encoding': 'gzip, deflate, sdch, br',
               'Accept-Language': 'zh-CN,zh;q=0.8',
               'Cache-Control': 'max-age=0',
               'Connection': 'keep-alive',
               'DNT': 1,
               'Host': 'www.baidu.com',
               'Upgrade-Insecure-Requests': 1,
               'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}
    query_link = 'https://www.baidu.com/s?wd='
    word1_url_encoded = urllib.quote(word1.encode('utf-8'))
    word2_url_encoded = urllib.quote(word2.encode('utf-8'))
    response1 = requests.get(query_link + word1_url_encoded, headers=headers).text

    response2 = requests.get(query_link + word2_url_encoded, headers=headers).text

    response3 = requests.get(query_link + word1_url_encoded + ' ' + word2_url_encoded, headers=headers).text
    num1 = get_num_from_html(response1)
    num2 = get_num_from_html(response2)
    num3 = get_num_from_html(response3)
    if num1 == -1 or num2 == -1 or num3 == -1:
        print 'failed to get numbers'
        return []
    N = pow(10, 16)
    features.append(web_jaccard(num1, num2, num3))
    features.append(web_overlap(num1, num2, num3))
    features.append(web_dice(num1, num2, num3))
    features.append(web_pmi(num1, num2, num3, N))
    return features


# 计算所有特征，列表形式返回，顺序：web-jaccard, web-overlap, web-dice, web-pmi，是否子串，共享字数
def get_all_features(word1, word2):
    features = get_four_features(word1, word2)
    if len(features) == 0:
        return []
    features.append(is_sub(word1, word2))
    features.append(get_shared_cha_num(word1, word2))
    return features


# 按照value值返回label
def get_label(value):
    # if value > 5:
    #     return 1
    # else:
    #     return 0
    # # t = round(value)
    # # print 'value:'+str(value)+' => '+str(t)
    return round(value)


# 从文件中读取特征矩阵
def load_features(filename):
    infile = codecs.open(filename, 'r', 'utf-8')
    data = numpy.empty((0, 7))
    lines = infile.readlines()
    lines.remove(lines[0])
    for line in lines:
        words = line.strip().split('\t')
        if len(words)<2:
            break
        f = []
        for i in range(3, 10):
            f.append(string.atof(words[i].strip()))
        data = numpy.row_stack([data, f])
    infile.close()
    return data


# 使用40条样例数据训练得到逻辑回归模型
def train():
    from Post import post
    data = load_features(macro.CORPUS_DIR + '/features_40_new.txt')
    y = []
    score = post.read_score(macro.CORPUS_DIR + 'a.txt', 43)
    for s in score:
        y.append(get_label(s))

    model = LogisticRegression()
    label = numpy.asarray(y).reshape(len(y))
    data = preprocessing.normalize(data)
    data = preprocessing.scale(data)

    model.fit(data, label)
    return model


if __name__ == '__main__':
    # model = train()
    # weight = range(0, 10)
    # data2 = load_features(macro.CORPUS_DIR + '/features_extra.txt')
    # score2 = post.read_score(macro.CORPUS_DIR + '/trial_data_ws_50_submit.txt', 50)
    # pre2 = []
    # #按照计算加权和作为分数
    # pre = model.predict_proba(data2) * weight
    # for p in pre:
    #     sum = 0
    #     for i in p:
    #         sum += i
    #     pre2.append(sum)
    # print eval.spearman(score2, pre2)
    s1 = 'das'
    s2 = 'das'
    print s1 == s2
