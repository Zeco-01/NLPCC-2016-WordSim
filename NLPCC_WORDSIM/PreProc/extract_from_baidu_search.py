# encoding=UTF-8
"""
    @author: Zeco on 2016/7/5
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from Com import utils
from Com import macro
from bs4 import BeautifulSoup
import requests
import urllib
import os
import codecs


def search_co(word1, word2):
    word_url_encoded = urllib.quote(word1.encode('utf-8')) + urllib.quote(word2.encode('utf-8'))
    request_url = "http://www.baidu.com/s?wd=" + word_url_encoded
    out_file_name = (macro.DICT_DIR + '/filter/' + word1 + '_' + word2 + '.txt').encode('GBK')
    # 文件已经存在，说明已经爬取过，直接跳过
    if os.path.isfile(out_file_name):
        return 0
    print word1, word2
    outfile = codecs.open(out_file_name, 'w', 'utf-8')
    try:
        response = requests.get(request_url)
    except requests.exceptions.RequestException:
        outfile.close()
        print -1
        return -1
    soup = BeautifulSoup(response.text, 'html.parser')
    divs = soup.find_all('div', class_='c-abstract')
    if len(divs) > 0:
        for div in divs:
            text = div.text
            cutlist = '\s\t\f\r\n。！？ 　'.decode('utf-8')
            sens = utils.cut(cutlist, text)
            for s in sens:
                if s.find(word1) != -1 and s.find(word2) != -1:
                    outfile.write(s + '\r\n')
    else:
        outfile.close()
        print -2
        return -2
    outfile.close()


def search(word):
    word_url_encoded = urllib.quote(word.encode('utf-8'))
    request_url = "http://www.baidu.com/s?wd=" + word_url_encoded
    out_file_name = (macro.BDNEWS_DOCS_ORG_DIR + '/' + word + '.txt').encode('GBK')

    # 文件已经存在，说明已经爬取过，直接跳过
    if os.path.isfile(out_file_name):
        return 0
    print word
    outfile = codecs.open(out_file_name, 'w', 'utf-8')
    try:
        response = requests.get(request_url)
    except requests.exceptions.RequestException:
        outfile.close()
        print -1
        return -1
    soup = BeautifulSoup(response.text, 'html.parser')
    divs = soup.find_all('div', class_='c-abstract')
    if len(divs) > 0:
        for div in divs:
            text = div.text
            cutlist = '\s\t\f\r\n。！？ 　'.decode('utf-8')
            sens = utils.cut(cutlist, text)
            for s in sens:
                if s.find(word) != -1:
                    outfile.write(s + '\r\n')
    else:
        outfile.close()
        print -2
        return -2
    outfile.close()


def search_all(filename):
    infile = codecs.open(filename, 'r', 'utf-8')
    i = 0
    while True:
        word = infile.readline().strip()
        i += 1
        if i > 8000:
            break
        search(word)

    infile.close()


if __name__ == '__main__':
    idl, w1l, w2l, score, headline = utils.read2wordlist([(macro.CORPUS_DIR, '500_2.csv')])
    for word1, word2 in zip(w1l, w2l):
        search_co(word1, word2)
