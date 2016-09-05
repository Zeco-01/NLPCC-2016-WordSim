# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from gensim import corpora
from Com import macro
from Com import utils
import codecs
from gensim.models.word2vec import Word2Vec
import string
import numpy


def filter_lines(lines):
    result = []
    for line in lines:
        words = line.encode('utf-8').strip().split(' ')
        if len(words)>2:
            result.append(line)
    return result


def split_lines(lines):
    result = []
    for line in lines:
        words = line.encode('utf-8').strip().split(' ')

        result.append(words)

    return result


def convert(token2id):
    id2token = {}
    for kk, vv in token2id.items():
        id2token[vv] = kk
    return id2token


def get_distance_single(word, sen):
    ss = sen.strip().split(' ')
    i = 0
    index = 0
    for w in ss:
        if w == word:
            index = i
            break
        else:
            i += 1
    l = len(ss)
    i = 0
    dis_vec = []
    while i < l:
        dis_vec.append(0 - index + i)
        i += 1
    return dis_vec


def get_distance_vector(word1, word2, sens):
    dis_vecs = []
    for s in sens:
        temp = []
        temp.append(get_distance_single(word1, s))
        temp.append(get_distance_single(word2, s))
        dis_vecs.append(temp)
    return dis_vecs


def get_text(w1=None, w2=None):
    idl, w1l, w2l, score, headline = utils.read2wordlist([(macro.CORPUS_DIR, '500_2.csv')])
    ids = []
    scores = []
    text = []
    dis_vecs = []
    for idw,word1,word2,s in zip(idl,w1l,w2l,score):

        try:
            infile = codecs.open(macro.DICT_DIR + '/filter/' + word1 + '_' + word2 + '.txt',
                                 'r', 'utf-8')
        except:
            print word1, word2
        lines = infile.readlines()
        lines = filter_lines(lines)
        if len(lines) < 3:
            continue
        if w1 and w2:
            if word1 == w1 and word2 == w2:
                text.extend(split_lines(lines))
                scores.append(score)
                ids.append(idw)
                dis_vecs.extend(get_distance_vector(word1, word2, lines))
            else:
                continue
        else:
            text.extend(split_lines(lines))

            temp = [s] * len(lines)
            temp2 = [idw] * len(lines)
            scores.extend(temp)
            ids.extend(temp2)

            dis_vecs.extend(get_distance_vector(word1, word2, lines))
        infile.close()
    return text, scores, ids, dis_vecs


def numberize(texts):
    d, ids = get_all_dictionary()
    result = []
    for sen in texts:
        temp = []
        for word in sen:
            word = d.token2id[word.decode('utf-8')]
            temp.append(word)
        result.append(temp)
    return result, ids


def get_dictionary(word1, word2):
    text, scores, ids, dis_vecs = get_text(word1, word2)
    dictionary = corpora.Dictionary(text)
    dictionary.id2token = convert(dictionary.token2id)
    return dictionary


def get_all_dictionary():
    texts, scores, ids, dis_vecs = get_text()
    dictionary = corpora.Dictionary(texts)
    dictionary.id2token = convert(dictionary.token2id)
    return dictionary, ids


def get_glodern_score(word1, word2):
    infile = codecs.open(macro.CORPUS_DIR + '/500_2.csv', 'r', 'utf-8')
    lines = infile.readlines()[1:]
    for line in lines:
        words = line.strip().split('\t')
        if len(words) < 3:
            break
        if word1 == words[1] and word2 == words[2]:
            score = string.atof(words[-1])
            return score
    infile.close()
    return 0


def get_w2v_model(filename):
    model = Word2Vec.load_word2vec_format(filename, binary=True)
    return model


def get_round(list):
    result = []
    for l in list:
        result.append(round(l))
    return result


def load_data(part, n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    texts, scores, ids, dis_vecs = get_text()
    scores = get_round(scores)
    temp, ids = numberize(texts)
    l = len(temp)
    part_len = l/5
    all_set = [temp,scores]

    part11 = all_set[0][0:part_len]
    part12 = all_set[0][part_len:2*part_len]
    part13 = all_set[0][2*part_len:3*part_len]
    part14 = all_set[0][3*part_len:4*part_len]
    part15 = all_set[0][4*part_len:l]
    parts1=[part11,part12,part13,part14,part15]


    part21 = all_set[1][0:part_len]
    part22 = all_set[1][part_len:2*part_len]
    part23 = all_set[1][2*part_len:3*part_len]
    part24 = all_set[1][3*part_len:4*part_len]
    part25 = all_set[1][4*part_len:l]
    parts2=[part21,part22,part23,part24,part25]


    d_part1 = dis_vecs[0:part_len]
    d_part2 = dis_vecs[part_len:2*part_len]
    d_part3 = dis_vecs[2*part_len:3*part_len]
    d_part4 = dis_vecs[3*part_len:4*part_len]
    d_part5 = dis_vecs[4*part_len:l]
    d_parts = [d_part1,d_part2,d_part3,d_part4,d_part5]

    id_part1 = ids[0:part_len]
    id_part2 = ids[part_len:2*part_len]
    id_part3 = ids[2*part_len:3*part_len]
    id_part4 = ids[3*part_len:4*part_len]
    id_part5 = ids[4*part_len:l]
    id_parts = [id_part1,id_part2,id_part3,id_part4,id_part5]


    train_set = [[],[]]
    ids_train = []
    dis_vecs_train = []
    for i in range(0,5):
        if i!=part-1:
            train_set[0].extend(parts1[i])
            train_set[1].extend(parts2[i])
            ids_train.extend(id_parts[i])
            dis_vecs_train.extend(d_parts[i])

    test_set = [parts1[part-1],parts2[part-1]]
    dis_vecs_test = d_parts[part-1]
    ids_test = id_parts[part-1]

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        new_dis_vecs_train = []
        new_ids_train = []
        i = 0
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
                new_dis_vecs_train.append(dis_vecs_train[i])
                new_ids_train.append(ids[i])
            i += 1
        train_set = (new_train_set_x, new_train_set_y)
        dis_vecs_train = new_dis_vecs_train
        ids_train = new_ids_train
        del new_train_set_x, new_train_set_y, new_dis_vecs_train, new_ids_train

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    dis_vecs_valid = [dis_vecs_train[s] for s in sidx[n_train:]]
    ids_valid = [ids_train[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    dis_vecs_train = [dis_vecs_train[s] for s in sidx[:n_train]]
    ids_train = [ids_train[s] for s in sidx[:n_train]]
    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]
        dis_vecs_test = [dis_vecs_test[i] for i in sorted_index]
        ids_test = [ids_test[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
        dis_vecs_valid = [dis_vecs_valid[i] for i in sorted_index]
        ids_valid = [ids_valid[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
        dis_vecs_train = [dis_vecs_train[i] for i in sorted_index]
        ids_train = [ids_train[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test, dis_vecs_train, ids_train, dis_vecs_valid, ids_valid, dis_vecs_test, ids_test


if __name__ == '__main__':
    pass
