# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from pypinyin import pinyin, lazy_pinyin
import pypinyin
import codecs
from Com import macro

def load_sim_dict():
    dict_file = codecs.open(macro.SIM_DICT_PATH, 'r', 'utf-8')
    lines = dict_file.readlines()
    sim_dict = {}
    for line in lines:
        strs = line.strip().split(',')
        for s in strs:
            if '' == s:
                continue
            ss = set()
            for s2 in strs:
                if s2 == s:
                    continue
                else:
                    ss.add(s2)
            sim_dict[s] = ss

    dict_file.close()
    return sim_dict


def is_similar(word1, word2, sim_dict):
    if word1 in sim_dict.keys():
        if word2 in sim_dict[word1]:
            return True
    return False

def get_pattern_sim(word1, word2, sim_dict):
    same_count = 0
    sim_count = 0
    temp = word2
    if word1 == word2:
        same_count = len(word1)
    else:
        i = 0
        for c in word1:
            word1_sub = word1[:i]
            if word1_sub.find(c) != -1:
                continue
            if temp.find(c) != -1:
                same_count += 1
                temp.replace(c, '')
            else:
                for c2 in word2:
                    if is_similar(c, c2, sim_dict):
                        sim_count += 1
                        break
    same_sim = 2 * same_count * 1.0 / (len(word1) + len(word2))
    sim_sim = 2 * sim_count * 1.0 / (len(word1) + len(word2))
    return same_sim + sim_sim

def get_pinyin_sim(word1, word2):
    i = 0
    count = 0
    while i < len(word1) and i < len(word2):
        py1 = pinyin(word1[i], style=pypinyin.TONE2, heteronym=True)[0]
        for p1 in py1:
            is_contain = False
            py2 = pinyin(word2[i], style=pypinyin.TONE2, heteronym=True)[0]
            for p2 in py2:
                if p1 == p2:
                    count += 1
                    is_contain = True
                    break
            if is_contain:
                break
        i += 1
    pinyin_sim = 2 * count * 1.0 / (len(word1) + len(word2))
    return pinyin_sim

def get_seq_sim(word1, word2):
    i = 0
    count = 0
    for c in word2:
        pos = word1.find(c)
        if i == pos:
            count += 1
    seq_sim = 2 * count * 1.0 / (len(word1) + len(word2))
    return seq_sim

if __name__ == '__main__':
    pass
