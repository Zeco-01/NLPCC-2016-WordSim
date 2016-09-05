# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from Com import macro
import math

import codecs
import string


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
    if pq < 5:
        return 0
    return math.log((N * pq) / (p * q), 2) / (math.log(N, 2))


def get_nums(word1, word2):
    infile = codecs.open(macro.WORD_LIST_PATH + 'word_nums_golden.txt', 'r', 'utf-8')
    lines = infile.readlines()
    lines.remove(lines[0])
    nums = []
    for line in lines:
        words = line.strip().split('\t')
        if words[1] == word1 and words[2] == word2:
            nums.append(string.atof(words[3]))
            nums.append(string.atof(words[4]))
            nums.append(string.atof(words[5]))
            break

    infile.close()
    return nums


def get_web_features(word1, word2):
    nums = get_nums(word1, word2)
    features = []
    if len(nums) == 0:
        pass
    features.append(web_jaccard(nums[0], nums[1], nums[2]))
    features.append(web_overlap(nums[0], nums[1], nums[2]))
    features.append(web_dice(nums[0], nums[1], nums[2]))
    features.append(web_pmi(nums[0], nums[1], nums[2], macro.N))
    return features
