# encoding=UTF-8
"""
    @author: Zeco on 2016/6/29
    @email: zhancong002@gmail.com
    @step:
    @function:
"""

from bs4 import BeautifulSoup
import requests
import urllib
import os
import codecs
import string
import functions


def get_all_articles(word, failed_urls):
    folder_name = word.encode("GBK")
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    word_url_encoded = urllib.quote(word.encode('utf-8'))
    request_url = "http://news.baidu.com/ns?word=" + word_url_encoded + "&tn=news&from=news&cl=2&rn=50&ct=1"
    response = requests.get(request_url)
    soup = BeautifulSoup(response.text, 'html5lib')
    result_heads = soup.find_all('h3')
    i = 0
    for head in result_heads:
        a_temp = head.contents[0]
        link = ''
        try:
            link = a_temp.attrs['href']
        except AttributeError:
            print a_temp
            continue
        else:
            pass
        title = ""
        print i
        i += 1
        for child in a_temp.children:
            if type(child) == type(a_temp):
                title += child.text
            else:
                title += child
        try:
            article = requests.get(link, timeout=5)
        except requests.exceptions.RequestException:
            failed_urls.append(link)
            print 'failed:' + link + str(len(failed_urls))
            continue
        else:
            pass
        if article.encoding == 'ISO-8859-1':
            encodings = requests.utils.get_encodings_from_content(article.content)
            if len(encodings) == 0:
                article.encoding = 'utf-8'
            else:
                article.encoding = encodings[0]
        article = article.text.encode('utf-8')
        title = title.replace('"', '')
        title = title.replace('/', '')
        title = title.replace('\\', '')
        title = title.replace('|', '')
        title = title.replace('?', '')
        title = title.replace(':', '')
        title = title.replace('*', '')
        title = title.replace('<', '')
        title = title.replace('>', '')
        try:
            title_gbk = title.encode("GBK")
        except UnicodeEncodeError:
            title_gbk = str(i)
        else:
            pass
        outfile = open(folder_name + "/" + title_gbk + ".txt", "w")
        outfile.write(article)
        outfile.close()
    return


def extract_sentences(word):
    folder_name = word.encode("GBK")
    list_dirs = os.walk(folder_name)
    i = 0
    for root, dirs, files in list_dirs:
        outfile = open('Extraction/' + folder_name + '.txt', 'w')
        for f in files:
            file_path = os.path.join(root, f)
            infile = codecs.open(file_path, 'r', 'utf-8')
            lines = infile.readlines()
            full_text = ''
            for line in lines:
                full_text += line.strip()
            infile.close()
            bs = BeautifulSoup(full_text, "lxml")
            for tag in bs(['script', 'img', 'a', 'head', 'li', 'style']):
                tag.extract()
            full_text = bs.get_text()
            cut_list = '\s\t\f\r\n。！？ 　'.decode('utf-8')
            sentences = functions.cut(cut_list, full_text)
            for s in sentences:
                if string.find(s, word) != -1:
                    i += 1
                    print 's:' + str(i)
                    s = s.encode('utf-8')
                    outfile.write(s + '\n')
        outfile.close()
    return


def get_extracted_sentences(file_name):
    outfile = codecs.open(file_name.decode('utf-8').encode('GBK'), 'r', 'utf-8')
    word = outfile.readline().strip()
    print word
    failed_urls = []
    while word != '\n':
        get_all_articles(word, failed_urls)
        extract_sentences(word)
        word = outfile.readline().strip()
        print word
    return
