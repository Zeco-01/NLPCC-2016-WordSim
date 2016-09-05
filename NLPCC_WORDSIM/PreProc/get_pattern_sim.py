# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
import similar_char


def get_pattern_sim(word1, word2,sim_dict):
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
                    if similar_char.is_similar(c, c2, sim_dict):
                        sim_count += 1
                        break
    same_sim = 2 * same_count * 1.0 / (len(word1) + len(word2))
    sim_sim = 2 * sim_count * 1.0 / (len(word1) + len(word2))
    return same_sim + sim_sim

if __name__ == '__main__':
    sim_dict = similar_char.load_sim_dict()
    print get_pattern_sim(u'只管',u'尽管',sim_dict)