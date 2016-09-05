# encoding=UTF-8
"""
    @author: Zeco on 2016/7/3
    @email: zhancong002@gmail.com
    @step:
    @function:一些工具性质的函数
"""
import re
from bs4 import BeautifulSoup


def find_token(cut_list, char):
    if char in cut_list:
        return True
    else:
        return False


def join(l):
    text = ''
    for i in l:
        text += i
    return text


def cut(cut_list, text):
    l = []
    sentences = []
    for char in text:
        if find_token(cut_list, char):
            sentence = join(l)
            pattern = re.compile(r'\s+')
            if not pattern.match(sentence):  # and len(sentence) > 4:
                l.append(char)
                sentences.append(join(l))
                l = []
            else:
                l = []
        else:
            l.append(char)
    return sentences


# 去除无用的标签
def remove_useless_tags(html):
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'img', 'a', 'head', 'li', 'style']):
        tag.extract()
    return soup.get_text()
