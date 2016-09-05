# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from Com import macro
import codecs


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


if __name__ == '__main__':
    sim_dict = load_sim_dict()
