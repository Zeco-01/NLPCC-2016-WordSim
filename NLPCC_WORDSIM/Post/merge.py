# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
import codecs
import string
import math
from Com import macro


# 合并/选择策略
def select(vv, vc, mode):
    if mode == macro.VECTOR_ONLY:
        return vv

    elif mode == macro.CILIN_ONLY:
        return vc

    elif mode == macro.AVERAGE:
        return (vc + vv) / 2.0

    elif mode == macro.MAX:
        return max([vv, vc])

    elif mode == macro.MIN:
        return min([vv, vc])

    elif mode == macro.REPLACE_1_AND_10:
        if vc == 1 or vc == 10.0:
            return vv
        return vc

    elif mode == macro.REPLACE_AND_AVERAGE:
        if vc == 1 or vc == 10.0:
            return vv
        else:
            return (vc + vv) / 2.0

    elif mode == macro.GEOMETRIC_MEAN:
        return math.sqrt(vv * vv + vc * vc)

    elif mode == macro.REPLACE_1:
        if vc == 1:
            return vv
        return vc
    elif mode == macro.REPLACE_1_AND_AVERAGE:
        if vc == 1:
            return vv
        else:
            return (vc + vv) / 2.0
    elif mode == macro.REPLACE_1_AND_GEOMETRIC_MEAN:
        if vc == 1:
            return vv
        else:
            return math.sqrt(vv * vc)
    elif mode == macro.REPLACE_1_AND_MIN:
        if vc == 1:
            return vv
        else:
            return min([vc, vv])


def merge_2_list(file_vec, file_ci, mode):
    f_vec = codecs.open(file_vec, 'r', 'utf-8')
    f_ci = codecs.open(file_ci, 'r', 'utf-8')
    f_vec.readline()
    f_ci.readline()
    values = []
    i = 0
    while True:
        i += 1
        linev = f_vec.readline()
        linec = f_ci.readline()
        wordsv = linev.strip().split('\t')
        wordsc = linec.strip().split('\t')
        if len(wordsc) < 4:
            break
        try:
            valuev = string.atof(wordsv[-1].strip())
            valuec = string.atof(wordsc[-1].strip())
            values.append(select(valuev, valuec, mode))
        except:
            pass
    f_ci.close()
    f_vec.close()
    return values


# 按照select的策略合并两个结果
def merge(file_vec, file_ci, file_out, mode):
    f_vec = codecs.open(file_vec, 'r', 'utf-8')
    f_ci = codecs.open(file_ci, 'r', 'utf-8')
    outfile = codecs.open(file_out, 'w', 'utf-8')
    f_vec.readline()
    f_ci.readline()
    values = []
    i = 0
    outfile.write('ID\tWord1\tWord\tSimilarity Score\r\n')
    while True:
        i += 1
        linev = f_vec.readline()
        linec = f_ci.readline()

        wordsv = linev.strip().split('\t')
        wordsc = linec.strip().split('\t')
        if len(wordsv) < 4:
            break
        print i
        valuev = string.atof(wordsv[-1].strip())
        valuec = string.atof(wordsc[-1].strip())
        word1 = wordsc[1].strip()
        word2 = wordsc[2].strip()
        print >> outfile, str(i) + '\t' + word1 + '\t' + word2 + '\t' + str((select(valuev, valuec, mode))) + '\r\n',
    outfile.close()
    f_ci.close()
    f_vec.close()
    return file_out


if __name__ == '__main__':
    merge(macro.RESULTS_DIR + '/fml_google_en_w2v.result',
          macro.RESULTS_DIR + '/evatestdata3_cilin.txt',
          macro.RESULTS_DIR + '/best_without_lstm.txt', macro.MAX)
    pass
