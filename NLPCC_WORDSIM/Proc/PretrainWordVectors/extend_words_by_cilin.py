# encoding=UTF-8
"""
    @author: Administrator on 2016/6/23
    @email: ppsunrise99@gmail.com
    @step:
    @function: 为了扩充更多的词义信息，训练词向量，可以利用cilin将同义词集扩充到词典中
"""
from Com import macro, utils


def dry_extend_vocab_by_cilin(f_tuple_list, fdir, fdict):
    # 语料中的词
    word1_list, word2_list, manu_sim_list, headline = utils.read2wordlist(f_tuple_list, mode='tag')
    vocab = sorted(list(set(word1_list + word2_list)))
    with open(r'%s/%s' % (macro.DICT_DIR, macro.DRY_ORG_VOCAB_DICT), 'w') as fw:
        new_vocab = '\n'.join(vocab) + '\n'
        fw.write(new_vocab.encode('utf-8'))

    cilin_list = utils.read_cilin2list()  # 得到长度在[1,4]的词林二维列表
    # 如果词林中的词在原始vocab中，那么就把这组近义词中加入vocab
    for row in cilin_list:
        for col in row:
            if col in vocab:
                print '%s in [%s]' % (col, ','.join(vocab))
                vocab.extend(row)  # 如果某个词出现在vocab中，就把这行的词都加入
                break  # 直接对下一行处理

    vocab = sorted(list(set(vocab)))  # 去重
    with open(r'%s/%s' % (fdir, fdict), 'w') as fw:
        ext_new_vocab = '\n'.join(vocab) + '\n'
        fw.write(ext_new_vocab.encode('utf-8'))

    return


def fml_extend_vocab_by_cilin(f_tuple_list, fdir, fdict):
    # 语料中的词
    id_list, word1_list, word2_list, headline = utils.read2wordlist(f_tuple_list, mode='no_tag')
    vocab = list(set(word1_list + word2_list))
    with open(r'%s/%s' % (macro.DICT_DIR, macro.FML_ORG_VOCAB_DICT), 'w') as fw:
        new_vocab = '\n'.join(vocab) + '\n'
        fw.write(new_vocab.encode('utf-8'))
    cilin_list = utils.read_cilin2list()  # 得到长度在[1,4]的词林二维列表
    # 如果词林中的词在原始vocab中，那么就把这组近义词中加入vocab
    for row in cilin_list:
        for col in row:
            if col in vocab:
                print '%s in [%s]' % (col, ','.join(vocab))
                vocab.extend(row)  # 如果某个词出现在vocab中，就把这行的词都加入
                vocab = list(set(vocab))  # 去重
                break  # 直接对下一行处理

    with open(r'%s/%s' % (fdir, fdict), 'w') as fw:
        ext_new_vocab = '\n'.join(vocab) + '\n'
        fw.write(ext_new_vocab.encode('utf-8'))

    return


# 样例数据
def dry_run():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]
    dry_extend_vocab_by_cilin(f_tuple_list, macro.DICT_DIR, macro.DRY_EXT_VOCAB_DICT)
    print 'dry run finished!'


# 正式提交数据
def formal_run():
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]
    fml_extend_vocab_by_cilin(f_tuple_list, macro.DICT_DIR, macro.FML_EXT_VOCAB_DICT)
    print 'formal run finished!'
    pass


if __name__ == '__main__':
    # dry_run()
    formal_run()
