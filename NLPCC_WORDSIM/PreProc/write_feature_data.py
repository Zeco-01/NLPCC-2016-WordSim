# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:写特征文件
"""
from Proc.LR import sk_LR
from Com import utils
from Com import macro
import codecs
import time
import similar_char
import get_pattern_sim
import get_seq_sim
import get_pinyin_sim
import get_web_features


# 计算并记录特征数据，num表示从多少条开始
def write_data(word_list_file, num, outfilename):
    infile = codecs.open(word_list_file, 'r', 'utf-8')
    outfile = codecs.open(outfilename, 'w', 'utf-8')
    line = infile.readline()
    outfile.write('ID\tWord1\tWord2\tweb-jaccard\tweb-overlap\tweb-dice\tweb-pmi\tIsSubString\tSharedCharNum\r\n')
    i = 0
    while i < num:
        infile.readline()
        i += 1
    while line != '\n':
        line = infile.readline()
        cutlist = ',\t\r'.decode('utf-8')
        i += 1
        words = utils.cut(cutlist, line)
        if len(words) == 0:
            break

        word1 = words[0].strip()
        word2 = words[1].strip()
        features = sk_LR.get_all_features(word1, word2)
        if i % 100 == 0:
            time.sleep(1)
        if len(features) == 0:
            print i
            break
        outfile.write(word1 + '\t' + word2 + '\t')
        for f in features:
            outfile.write(str(f) + '\t')
        outfile.write('\r\n')
    outfile.close()
    infile.close()


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
    sim_dict = similar_char.load_sim_dict()
    for wp in word_pairs:
        features = []
        web_features = get_web_features.get_web_features(wp.word1, wp.word2)
        features.extend(web_features)
        features.append(get_pinyin_sim.get_pinyin_sim(wp.word1, wp.word2))
        features.append(get_seq_sim.get_seq_sim(wp.word1, wp.word2))
        features.append(get_pattern_sim.get_pattern_sim(wp.word1, wp.word2, sim_dict))
        out_file.write(fea_2_line(wp.id, wp.word1, wp.word2, features))
    out_file.close()


if __name__ == '__main__':
    write_features(macro.CORPUS_DIR + '/500_2.csv')
