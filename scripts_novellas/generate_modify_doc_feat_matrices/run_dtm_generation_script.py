system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os

from preprocessing.presetting import global_corpus_representation_directory, global_corpus_directory, language_model_path, vocab_lists_dicts_directory, word_translate_table_to_dict
from preprocessing.corpus import DTM


outfile_dtm_path = os.path.join(global_corpus_representation_directory(system), "raw_dtm")
normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")
corpus_path = global_corpus_directory(system, test=False)
my_model_de_path = language_model_path(system)
n_mfw = 6000


normalization_dict, dict_lower = word_translate_table_to_dict(normalization_table_path)
use_idf = True
normalization = "l1"

raw_corpus_object = DTM(corpus_path=corpus_path, dehyphen=True, lemmatize=False,
                            translate_umlaute=False, inverse_translate_umlaute=False,
                            sz_to_ss=False,
                            normalize_orthogr=True, normalization_table_path=normalization_table_path,
                            correct_ocr=True, eliminiate_pagecounts=True, n_mfw=n_mfw,
                            eliminate_pos_items=True, norm=normalization, language_model=my_model_de_path, use_idf=use_idf)
raw_corpus_object.generate_corpus_as_dict()


normalization_list = ["l1", "l2"]

use_idf = True
for normalization in normalization_list:
    new_corpus_object = raw_corpus_object
    new_corpus_object.norm = normalization
    new_corpus_object.use_idf = use_idf

    new_corpus_object.generate_dtm_from_dict()
    outfile_basename = "raw_dtm_"+str(normalization)+"_"+"_use_idf_"+str(use_idf)+str(n_mfw)+"mfw.csv"
    dtm_outfile_path = os.path.join(outfile_dtm_path, outfile_basename)
    new_corpus_object.save_csv(dtm_outfile_path)

use_idf = False
for normalization in normalization_list:
    new_corpus_object = raw_corpus_object
    new_corpus_object.norm = normalization
    new_corpus_object.use_idf = use_idf

    new_corpus_object.generate_dtm_from_dict()
    outfile_basename = "raw_dtm_"+str(normalization)+"_"+"_use_idf_"+str(use_idf)+str(n_mfw)+"mfw.csv"
    dtm_outfile_path = os.path.join(outfile_dtm_path, outfile_basename)
    new_corpus_object.save_csv(dtm_outfile_path)