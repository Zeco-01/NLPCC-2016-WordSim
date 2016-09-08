# encoding=UTF-8
"""
    @author: Zeco on 2016/6/29
    @email: zhancong002@gmail.com
    @step:
    @function:爬取写搜网例句
"""
from Com import macro
from Com import utils
import codecs
import urllib
import requests
import string
import threading
import os

"""
说明：目前本代码只能爬取写搜网的例句
"""


# 从获取到的html中抽取句子
def extract_from_html(word, html, pagenum):
    text = utils.remove_useless_tags(html)
    cut_list = '!?.\t\r\n。！？'.decode('utf-8')
    # 含有关键词的句子列表
    sentences = []
    for s in utils.cut(cut_list, text):
        if string.find(s, word) != -1:
            sentences.append(s)
    if len(sentences) > 0:
        sentences.remove(sentences[0])
    return sentences


# 根据关键词和页码获取写搜网的对应搜索结果列表html
def get_html_by_word(word, pagenum):
    word_url_encoded = urllib.quote(word.encode('gb2312'))
    # 判断页码
    if pagenum == 0:
        link = 'http://juzi.xieso.net/' + word_url_encoded + '/'
    else:
        link = 'http://juzi.xieso.net/' + word_url_encoded + '/list' + str(pagenum) + '.html'
    try:
        original_html = requests.get(link).text
    except requests.exceptions.RequestException:
        return ''
    return original_html


def extract(word_file, quater_no, total_word_num, thread_num):
    infile = codecs.open(word_file, 'r', 'utf-8')
    failed_file_name = (macro.DICT_DIR + '/failed' + str(quater_no) + '.txt').encode('GBK')
    failed_file = codecs.open(failed_file_name, 'w', 'utf-8')
    quater = total_word_num / thread_num
    i = (quater_no - 1) * quater + 1
    j = 1
    num = 0
    while j < i:
        infile.readline()
        j += 1
    while i < quater_no * quater and i > (quater_no - 1) * quater:
        try:
            word = infile.readline().strip()
        except BaseException as err:
            print err
            continue
        outfile_name = (macro.XIESO_DOCS_ORG_DIR + '/' + word + '.txt').encode('GBK')
        # 防止重复爬取
        if os.path.isfile(outfile_name):
            return 0
        outfile = codecs.open(outfile_name, 'w', 'utf-8')
        # 计数
        num += 1
        print num
        pagenum = -1
        while True:
            pagenum += 1
            original_html = get_html_by_word(word, pagenum)
            if len(original_html) > 0:
                sentences = extract_from_html(word, original_html, pagenum)
                for s in sentences:
                    outfile.write(s + '\r\n')
            else:
                failed_file.write(word + '\r\n')
                break
            # 小于29行说明已经是最后一页
            if len(sentences) < 29:
                break
        outfile.close()
    infile.close()
    failed_file.close()
    return 1


class extract_thread(threading.Thread):
    def __init__(self, word_file, quater_no, total_word_num, thread_num):
        threading.Thread.__init__(self)
        self.word_file = word_file
        self.total_word_num = total_word_num
        self.quater_no = quater_no
        self.thread_num = thread_num

    def run(self):
        extract(self.word_file, self.quater_no, self.total_word_num, self.thread_num)


def run_e(word_file):
    infile = codecs.open(word_file, 'r', 'utf-8')
    lines = infile.readlines()
    total_num = len(lines)
    thread_num = 4
    i = 0
    while i < thread_num:
        new_thread = extract_thread(word_file, i + 1, total_num, thread_num)
        new_thread.start()
        i += 1
    return


if __name__ == '__main__':
    run_e(macro.DICT_DIR + '/dry_origin_vocab.txt')
