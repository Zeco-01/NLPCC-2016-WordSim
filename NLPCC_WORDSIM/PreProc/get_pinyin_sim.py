# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from pypinyin import pinyin, lazy_pinyin
import pypinyin


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


if __name__ == '__main__':
    i = get_pinyin_sim(u'没戏', u'没辙')
    print i
