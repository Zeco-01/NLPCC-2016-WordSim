# encoding=UTF-8
"""
    @author: Zeco on 
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from Com import macro
from Eval import eval
from Com import utils
import post

dry_results = [macro.DRY_DATATANG_RESULT,
               macro.DRY_WIKI_RESULT,
               macro.DRY_ORG_BDNEWS_RESULT,
               macro.DRY_EXT_BDNEWS_RESULT,
               macro.DRY_ORG_XIESO_RESULT,
               macro.DRY_EXT_XIESO_RESULT,
               macro.DRY_ORG_BDNEWS_XIESO_RESULT,
               macro.DRY_EXT_BDNEWS_XIESO_RESULT,
               macro.DRY_ORG_BDNEWS_XIESO_DATATANG_RESULT,
               macro.DRY_EXT_BDNEWS_XIESO_DATATANG_RESULT,
               macro.DRY_ORG_BDNEWS_XIESO_WIKI_RESULT,
               macro.DRY_EXT_BDNEWS_XIESO_WIKI_RESULT,
               macro.DRY_ORG_BDNEWS_XIESO_DATATANG_WIKI_RESULT,
               macro.DRY_EXT_BDNEWS_XIESO_DATATANG_WIKI_RESULT
               ]
formal_results = [macro.FML_DATATANG_RESULT,
                  macro.FML_WIKI_RESULT,
                  macro.FML_ORG_BDNEWS_RESULT,
                  macro.FML_EXT_BDNEWS_RESULT,
                  macro.FML_ORG_XIESO_RESULT,
                  macro.FML_EXT_XIESO_RESULT,
                  macro.FML_ORG_BDNEWS_XIESO_RESULT,
                  macro.FML_EXT_BDNEWS_XIESO_RESULT,
                  macro.FML_ORG_BDNEWS_XIESO_DATATANG_RESULT,
                  macro.FML_EXT_BDNEWS_XIESO_DATATANG_RESULT,
                  macro.FML_ORG_BDNEWS_XIESO_WIKI_RESULT,
                  macro.FML_EXT_BDNEWS_XIESO_WIKI_RESULT,
                  macro.FML_ORG_BDNEWS_XIESO_DATATANG_WIKI_RESULT,
                  macro.FML_EXT_BDNEWS_XIESO_DATATANG_WIKI_RESULT]


def compare():
    formal_pred_all_features = post.get_value_list(macro.CORPUS_DIR + '/features_golden_new.txt', [1, 1, 1, 1, 1, 1, 1])
    formal_pred_selected_features = post.get_value_list(macro.CORPUS_DIR + '/features_golden_new.txt',
                                                        [0, 0, 0, 1, 0, 1, 0])
    dry_pred_all_features = post.get_value_list(macro.CORPUS_DIR + '/features_test.txt', [1, 1, 1, 1, 1, 1, 1])
    dry_pred_selected_features = post.get_value_list(macro.CORPUS_DIR + '/features_test.txt', [0, 0, 0, 1, 0, 1, 0])
    for result in dry_results:
        idl, w1l, w2l, scores, headline = utils.read2wordlist([(macro.RESULTS_DIR, result)])
        print str(result) + ' vs dry_pred_all_featuers spearman: ', eval.spearman(dry_pred_all_features, scores)[
            0], 'pearson: ', eval.pearson(dry_pred_all_features, scores)[0]

        print str(result) + ' vs dry_pred_selected_featuers spearman: ', eval.spearman(dry_pred_selected_features,
                                                                                       scores)[0], 'pearson: ', \
            eval.pearson(dry_pred_selected_features, scores)[0]

    for result in formal_results:
        idl, w1l, w2l, scores, headline = utils.read2wordlist([(macro.RESULTS_DIR, result)])
        print str(result) + ' vs formal_pred_all_featuers spearman: ', eval.spearman(formal_pred_all_features, scores)[
            0], 'pearson: ', eval.pearson(formal_pred_all_features, scores)[0]

        print str(result) + ' vs formal_pred_selected_featuers spearman: ', eval.spearman(formal_pred_selected_features,
                                                                                          scores)[0], 'pearson: ', \
            eval.pearson(formal_pred_selected_features, scores)[0]


if __name__ == '__main__':
    compare()
