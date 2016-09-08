#!usr/bin/env python
# -*- coding:utf-8 -*-

from Com import macro
from nltk.corpus import wordnet as wn

import pandas as pd
import numpy as np
from scipy import stats

cmn = 'cmn'

data = pd.read_csv('%s/%s' % (macro.CORPUS_DIR, 'NLPCC_Formal500.csv'), encoding='utf-8')
wordsList = np.array(data.iloc[:, [1, 2]])
simScore = np.array(data.iloc[:, [3]])
idList = data.iloc[:, [0]]
default_sim = 1
predScoreList = []
for i, (word1, word2) in enumerate(wordsList):
    try:
        synsets1 = wn.synsets(word1.decode('utf-8'), lang=cmn)
        synsets2 = wn.synsets(word2.decode('utf-8'), lang=cmn)
        sim_tmp = []
        for synset1 in synsets1:
            for synset2 in synsets2:
                score = synset1.path_similarity(synset2)
                # score = synset1.wup_similarity(synset2)
                # score = synset1.lch_similarity(synset2)
                # score = synset1.lin_similarity(synset2)
                # score = synset1.res_similarity(synset2)
                # score = synset1.jcn_similarity(synset2)
                if score is None:
                    score = 0
                sim_tmp.append(score)
                # print word1, ',', word2, 'score:::', score
        if sim_tmp:
            auto_sim = np.mean(sim_tmp) * 9 + 1
        else:
            auto_sim = np.nan
    except:
        auto_sim = np.nan
        print 'word is not in list'

    predScoreList.append(auto_sim)
    print "process #%d words pair [%s,%s] %s %s" % (i, word1, word2, simScore[i], auto_sim)

impMmsList = np.array(predScoreList).reshape((len(predScoreList), 1))
(coef1, pvalue) = stats.spearmanr(simScore, impMmsList)
print coef1
# (correlation=0.3136469783526708, pvalue=1.6943792485183932e-09)


submitData = np.hstack((idList, wordsList, simScore, impMmsList))
(pd.DataFrame(submitData)).to_csv("%s/wordnet.csv" % macro.RESULTS_DIR, index=False, encoding='gbk',
                                  header=["ID", "Word1", "Word2", "Score", "Prediction"])

if __name__ == '__main__':
    pass
