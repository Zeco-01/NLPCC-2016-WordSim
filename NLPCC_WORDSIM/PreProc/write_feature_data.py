# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:写特征文件
"""

from Com import utils
from Com import macro
import codecs
import get_sims
import get_web_features


def fea_2_line(wpid, word1, word2, features):
    line = wpid + '\t' + word1 + '\t' + word2 + '\t'
    for f in features:
        line += str(f) + '\t'
    line += '\r\n'
    return line


def write_features(word_list_file_name):
    word_pairs = utils.read_word_list(word_list_file_name)
    out_file = codecs.open(macro.WORD_LIST_PATH + 'features_golden_new.txt', 'w', 'utf-8')
    out_file.write(
        'ID\tWord1\tWord2\tweb-jaccard\tweb-overlap\tweb-dice\tweb-pmi\tpinyin_sim\tseq_sim\tpattern_sim\t\r\n')
    sim_dict = get_sims.load_sim_dict()
    for wp in word_pairs:
        features = []
        web_features = get_web_features.get_web_features(wp.word1, wp.word2)
        features.extend(web_features)
        features.append(get_sims.get_pinyin_sim(wp.word1, wp.word2))
        features.append(get_sims.get_seq_sim(wp.word1, wp.word2))
        features.append(get_sims.get_pattern_sim(wp.word1, wp.word2, sim_dict))
        out_file.write(fea_2_line(wp.id, wp.word1, wp.word2, features))
    out_file.close()


if __name__ == '__main__':
    write_features(macro.CORPUS_DIR + '/500_2.csv')
