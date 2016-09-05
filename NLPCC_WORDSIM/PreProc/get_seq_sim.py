# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""


def get_seq_sim(word1, word2):
    i = 0
    count = 0
    for c in word2:
        pos = word1.find(c)
        if i == pos:
            count += 1
    seq_sim = 2 * count * 1.0 / (len(word1) + len(word2))
    return seq_sim
