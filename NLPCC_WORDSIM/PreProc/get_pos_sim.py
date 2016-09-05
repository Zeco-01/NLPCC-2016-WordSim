# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
# TO DO
import jieba.posseg as pseg


def get_pos_sim(word1, word2):
    words1 = pseg.cut(word1)
    words2 = pseg.cut(word2)


if __name__ == '__main__':
    get_pos_sim(u'天空不留下鸟的痕迹', u'但我已飞过')
