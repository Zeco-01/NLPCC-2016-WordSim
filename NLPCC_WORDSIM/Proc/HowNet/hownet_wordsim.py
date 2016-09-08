# encoding=UTF-8
"""
    @author: Administrator on 2016/6/23
    @email: ppsunrise99@gmail.com
    @step:
    1、从评测文件读入wordlist；
    2、prepare_hownet_input转化成HowNet程序读入格式，注意编码格式必须是ANSI=GB2312
    3、到客户端依次点击【文件：输入】->【文件：查看词语义项】->【文件：查看词语义相似度】
    4、将输出结果转化成我们的格式，然后评价
    @function: 利用hownet计算相似度
    @question: 客户端界面中直接计算对于未登录词返回相似度是0，可是文件输入时未登录词返回相似度是1
"""
from Eval import eval
from Com import macro, utils
import numpy as np
import os
import codecs


class HowNetSim:
    def __init__(self, f_tuple_list, mode='tag'):
        self.f_tuple_list = f_tuple_list
        if mode == 'tag':
            self.id_list, self.word1_list, self.word2_list, self.manu_sim_list, self.headline = utils.read2wordlist(
                f_tuple_list, mode)
        elif mode == 'no_tag':
            self.id_list, self.word1_list, self.word2_list, self.headline = utils.read2wordlist(f_tuple_list, mode)

        self.ofname = '_'.join([f_tuple[1].split('.')[0] for f_tuple in self.f_tuple_list]) + '_hownet.txt'  # 输出文件名

    # 1.将我们的输入文件转成HowNet输入文件格式
    def prepare_hownet_input(self):
        fw = open('%s/%s' % (macro.HOWNET_DIR, 'TestWords.Txt'), 'w')  # HowNet程序的输入文件
        for w1, w2 in zip(self.word1_list, self.word2_list):
            fw.write('%s\n%s\n\n' % (w1.encode('GB2312'), w2.encode('GB2312')))
        fw.close()

    # 2. 客户端查询词对的义项     3.筛选掉没有搜索到义项的词对
    def filter(self):
        # 读义项文件并将其重命名为.Old
        meaning_path = '%s/%s' % (macro.HOWNET_DIR, 'TestMeanings.Txt')
        old_meaning_path = meaning_path + '.Old'
        # 如果上步骤得到的备份旧文件存在，则读取备份文件
        if os.path.exists(old_meaning_path):
            fr = codecs.open(old_meaning_path, 'r')
            content = fr.read().strip().split('\n\n')
            fr.close()
        else:
            fr = codecs.open(meaning_path, 'r')
            content = fr.read().strip().split('\n\n')
            fr.close()
            os.renames(meaning_path, old_meaning_path)
        # 将筛选后可以计算的内容写到文件中
        fw = codecs.open(meaning_path, 'w')
        pair_items = [pair.split('\n') for pair in content]
        abnormal_tags = []
        for item in pair_items:
            item_len = len(item)
            if item_len < 1:
                abnormal_tags.append(True)
            else:
                abnormal_tags.append(False)
                fw.write('\n'.join(item) + '\n\n')
        fw.close()
        return abnormal_tags

    # 4. 客户端计算相似度         5.将HowNet输出文件转化成我们的文件格式
    def proc_hownet_output(self):
        self.prepare_hownet_input()
        fr = open('%s/%s' % (macro.HOWNET_DIR, 'TestSimilarities.Txt'), 'r')  # HowNet程序的输出文件
        content = fr.read().decode('GB2312').encode('UTF-8')
        fr.close()
        analysis_items = content.split('===================================================\n')[:-1]  # 去掉最后一空项
        results_items = [ana_item.split('\n')[-2] for ana_item in analysis_items]  # 跳过最后一个空项，倒数第二个
        fw = open('%s/%s' % (macro.RESULTS_DIR, self.ofname), 'w')
        fw.write(self.headline.strip() + '\tauto_sim_score\n')
        auto_sim_list = []
        for id, w1, w2, manu_sim, item in zip(self.id_list, self.word1_list, self.word2_list, self.manu_sim_list,
                                              results_items):
            word_pair, sim = item.strip().split(':')
            auto_sim = np.float(sim) * 9 + 1
            auto_sim_list.append(auto_sim)
            ww1, ww2 = word_pair.split(',')
            if (w1 == ww1 and w2 == ww2) or (w1 == ww2 and w2 == ww1):
                newline = '%s\t%s\t%s\t%s\t%s\n' % (id, w1, w2, manu_sim, auto_sim)
                fw.write(newline)
                print newline.strip()
            else:
                print 'Err:word has been changed!!!%s!=%s;%s!=%s' % (w1, ww1, w2, ww2)
        return self.word1_list, self.word2_list, self.manu_sim_list, auto_sim_list

    def formal_proc_hownet_output(self):
        self.prepare_hownet_input()
        fr = open('%s/%s' % (macro.HOWNET_DIR, 'TestSimilarities.Txt'), 'r')  # HowNet程序的输出文件
        content = fr.read().decode('GB2312').encode('UTF-8')
        fr.close()
        analysis_items = content.split('===================================================\n')[:-1]  # 去掉最后一空项
        results_items = [ana_item.split('\n')[-2] for ana_item in analysis_items]  # 跳过最后一个空项，倒数第二个
        fw = open('%s/%s' % (macro.RESULTS_DIR, self.ofname), 'w')
        fw.write(self.headline.strip() + '\tauto_sim_score\n')
        auto_sim_list = []
        for id, w1, w2, manu_sim, item in zip(self.id_list, self.word1_list, self.word2_list, self.manu_sim_list,
                                              results_items):
            word_pair, sim = item.strip().split(':')
            if np.float(sim) < 0:
                auto_sim = 1
            else:
                auto_sim = 9 * np.float(sim) + 1
            auto_sim_list.append(auto_sim)
            ww1, ww2 = word_pair.split(',')
            if (w1 == ww1 and w2 == ww2) or (w1 == ww2 and w2 == ww1):
                newline = '%s\t%s\t%s\t%s\t%s\n' % (id, w1, w2, manu_sim, auto_sim)
                fw.write(newline)
                print newline.strip()
            else:
                print 'Err:word has been changed!!!%s!=%s;%s!=%s' % (w1, ww1, w2, ww2)
        return self.word1_list, self.word2_list, auto_sim_list

    # 评价HowNet计算结果
    def evalutaion(self):
        print 'spearman', eval.evaluate(macro.RESULTS_DIR, self.ofname, mode='spearman')
        print 'pearson', eval.evaluate(macro.RESULTS_DIR, self.ofname, mode='pearson')
        return


if __name__ == '__main__':
    # dry run
    # f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_DRY_FILE)]
    # wordsim_obj = HowNetSim(f_tuple_list, 'tag')
    # wordsim_obj.prepare_hownet_input()
    # wordsim_obj.formal_proc_hownet_output()
    # wordsim_obj.evalutaion()

    # formal run 500有标记的
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_GD_FILE)]
    wordsim_obj = HowNetSim(f_tuple_list, 'tag')
    wordsim_obj.prepare_hownet_input()
    wordsim_obj.filter()
    # wordsim_obj.formal_proc_hownet_output()
    # wordsim_obj.evalutaion()
    print 'finished!'
