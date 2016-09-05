# encoding=UTF-8
"""
    @author: Zeco on 2016/6/27
    @email: zhancong002@gmail.com
    @step:
    @function:分词
"""
from Com import macro
import jieba
import codecs
import os


def seg(in_folder_name, out_folder_name):
    list_dirs = os.walk(in_folder_name)
    for root, dirs, files in list_dirs:
        for f in files:
            word = f.decode('GBK')[:-4]
            out_file_name = (out_folder_name.decode() + word + '.txt').encode('GBK')
            file_path = os.path.join(root, f)
            infile = codecs.open(file_path, 'r', 'utf-8')
            try:
                lines = infile.readlines()
            except UnicodeDecodeError:
                print word
                continue
            outfile = codecs.open(out_file_name, 'w', 'utf-8')
            for line in lines:
                seg_words = jieba.cut(line, cut_all=False)

                for w in seg_words:
                    outfile.write(w + ' ')
            outfile.close()
    return


if __name__ == '__main__':
    in_folder_name = (macro.DICT_DIR + '/filter/').encode('GBK')
    out_folder_name = (macro.DICT_DIR + '/filter/').encode('GBK')
    seg(in_folder_name, out_folder_name)
