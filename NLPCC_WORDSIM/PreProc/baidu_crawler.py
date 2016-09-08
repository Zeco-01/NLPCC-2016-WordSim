# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from bs4 import BeautifulSoup
from Com import macro
from Com import utils

import operator
import string
import requests
import codecs
import urllib
import os.path


# 从html中提取搜索结果数量 百度返回不超过一亿
def get_num_from_html(html):
    bs = BeautifulSoup(html, 'html.parser')
    divs = bs.find_all('div', class_='nums')
    t = ''
    if len(divs) > 0:
        div = divs[0]
        t = div.text
    else:
        return -1

    num = filter(operator.methodcaller('isdigit'), t)
    num = string.atof(num)

    return num


def crawl_num(word1, word2):
    nums = []
    print 'crawl_num: ' + word1 + ' ' + word2
    headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'Accept-Encoding': 'gzip, deflate, sdch, br',
               'Accept-Language': 'zh-CN,zh;q=0.8',
               'Cache-Control': 'max-age=0',
               'Connection': 'keep-alive',
               'DNT': 1,
               'Host': 'www.baidu.com',
               'Upgrade-Insecure-Requests': 1,
               'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}
    query_link = 'https://www.baidu.com/s?wd='
    word1_url_encoded = urllib.quote(word1.encode('utf-8'))
    word2_url_encoded = urllib.quote(word2.encode('utf-8'))
    response1 = requests.get(query_link + word1_url_encoded, headers=headers).text

    response2 = requests.get(query_link + word2_url_encoded, headers=headers).text

    response3 = requests.get(query_link + word1_url_encoded + ' ' + word2_url_encoded, headers=headers).text
    num1 = get_num_from_html(response1)
    num2 = get_num_from_html(response2)
    num3 = get_num_from_html(response3)
    nums.append(num1)
    nums.append(num2)
    nums.append(num3)
    if num1 == -1 or num2 == -1 or num3 == -1:
        print 'failed to get numbers'
        return []
    return nums


def merge_result():
    i = 0
    out_file = codecs.open(macro.WORD_LIST_PATH + 'word_nums.txt', 'w', 'utf-8')
    while i < 10000:
        infile_name = macro.WORD_LIST_PATH + 'word_nums_from' + str(i) + '.txt'
        if os.path.isfile(infile_name):
            infile = codecs.open(infile_name, 'r', 'utf-8')
            lines = infile.readlines()
            lines.remove(lines[0])
            for line in lines:
                out_file.write(line)
            infile.close()
        else:
            continue
    out_file.close()


def crawl_all(word_list_file, num=0):
    out_file = codecs.open(macro.WORD_LIST_PATH + '50_word_nums_from_' + str(num) + '.txt', 'w', 'utf-8')
    out_file.write('ID\tWord1\tWord2\tNum1\tNum2\tNum3\r\n')
    word_pairs = utils.read_word_list(word_list_file)
    for wp in word_pairs[num:]:
        nums = crawl_num(wp.word1, wp.word2)
        if 0 == len(nums):
            print 'Failed to get num: ' + wp.word1 + ' ' + wp.word2
            print 'id: ' + str(wp.id)
            break
        else:
            out_file.write(
                str(wp.id) + '\t' + wp.word1 + '\t' + wp.word2 + '\t' + str(nums[0]) + '\t' + str(nums[1]) + '\t' + str(
                    nums[2]) + '\r\n')
    out_file.close()
    merge_result()


if __name__ == '__main__':
    #crawl_all(macro.WORD_LIST_PATH + 'test2.txt')
    print 'test'