# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from Com import utils
from Com import macro
import codecs


def contains(list, word):
    for l in list:
        l2 = l.decode('utf-8')
        if l2 == word:
            return True
    return False


def list2sen(list):
    sen = ''
    for l in list:
        sen += l
        sen += ' '
    return sen.decode('utf-8') + '\r\n'



if __name__ == '__main__':
    sentences = []
    sens_baidu = utils.f_tuple_list2sens([(macro.CORPUS_DIR, '500_2.csv')], macro.BDNEWS_DOCS_SEG_DIR, '')
    sens_xieso = utils.f_tuple_list2sens([(macro.CORPUS_DIR, '500_2.csv')], macro.XIESO_DOCS_SEG_DIR, '')
    sentences.extend(sens_baidu)
    sentences.extend(sens_xieso)
    idl, w1l, w2l, score, headline = utils.read2wordlist([(macro.CORPUS_DIR, '500_2.csv')])
    i = 0
    for word1 in w1l:
        word2 = w2l[i]
        i += 1
        print i, ' of 500'
        outfile = codecs.open(macro.DICT_DIR + '/filter/' + word1 + '_' + word2 + '.txt', 'w', 'utf-8')
        for sen in sentences:
            if contains(sen, word1) and contains(sen, word2):
                outfile.write(list2sen(sen))
        outfile.close()
