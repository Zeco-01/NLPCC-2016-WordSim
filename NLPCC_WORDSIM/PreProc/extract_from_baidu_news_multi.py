# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:爬取百度新闻语料
"""
from Com import macro
from Com import utils
from bs4 import BeautifulSoup
import requests
import urllib
import os
import codecs
import string
import threading


def extract(word):
    out_file_name = (macro.BDNEWS_DOCS_ORG_DIR + '/' + word + '.txt').encode('GBK')

    # 文件已经存在，说明已经爬取过，直接跳过
    if os.path.isfile(out_file_name):
        return 0

    outfile = codecs.open(out_file_name, 'w', 'utf-8')
    word_url_encoded = urllib.quote(word.encode('utf-8'))
    # 向百度新闻发起搜索请求的URL
    request_url = "http://news.baidu.com/ns?word=" + word_url_encoded + "&tn=news&from=news&cl=2&rn=50&ct=1"
    try:
        response = requests.get(request_url)
    except requests.exceptions.RequestException:
        outfile.close()
        return -1
    soup = BeautifulSoup(response.text, 'html.parser')
    # 找到所有h3标签，读取内容
    result_heads = soup.find_all('h3')
    for head in result_heads:
        # 有时因为网络问题会导致返回的结果页面异常
        if head['class'] == 'norsTitle':
            return -1
        a_temp = head.contents[0]
        link = ''
        try:
            link = a_temp.attrs['href']
        except AttributeError:
            continue
        else:
            pass
        try:
            article = requests.get(link, timeout=3)
        except requests.exceptions.RequestException:
            print 'failed:' + link
            continue
        else:
            pass
        # 检测并转换编码
        if article.encoding == 'ISO-8859-1':
            encodings = requests.utils.get_encodings_from_content(article.content)
            if len(encodings) == 0:
                article.encoding = 'utf-8'
            else:
                article.encoding = encodings[0]
        article = article.text.encode('utf-8').decode('utf-8')
        full_text = utils.remove_useless_tags(article)
        # 分句
        cut_list = '\s\t\f\r\n。！？ 　'.decode('utf-8')
        sentences = utils.cut(cut_list, full_text)
        for s in sentences:
            # 含有关键词则写文件
            if string.find(s, word) != -1:
                outfile.write(s + '\r\n')
    outfile.close()
    return 1


class extract_thread_b(threading.Thread):
    def __init__(self, word_file, quater_no, total_word_num, thread_num):
        threading.Thread.__init__(self)
        self.word_file = word_file
        self.total_word_num = total_word_num
        self.quater_no = quater_no
        self.thread_num = thread_num

    def run(self):
        quater = self.total_word_num / self.thread_num
        infile = codecs.open(self.word_file, 'r', 'utf-8')
        failed_out_name = macro.DICT_DIR + '/failed_words_baidu_news' + str(self.quater_no) + '.txt'
        failed_file = codecs.open(failed_out_name, 'w', 'utf-8')
        i = (self.quater_no - 1) * quater + 1
        j = 1
        # 跳过不属于本进程的部分
        while j < i:
            infile.readline()
            j += 1
        while i < self.quater_no * quater and i > (self.quater_no - 1) * quater:
            if self.quater_no == 1:
                print i
            word = infile.readline().strip()
            print word
            result_code = extract(word)
            # 记录查找失败的词
            if result_code == -1:
                pass

            i += 1
            failed_file.close()


def extract_from_baidu_news(word_file):
    infile = codecs.open(word_file, 'r', 'utf-8')
    lines = infile.readlines()
    total_num = len(lines)
    thread_num = 4
    i = 0
    while i < thread_num:
        new_thread = extract_thread_b(word_file, i + 1, total_num, thread_num)
        new_thread.start()
        i += 1
    return

if __name__ == '__main__':
    extract_from_baidu_news(macro.DICT_DIR + '/fml_origin_vocab_half1.txt')
