system = "wcph113"
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
n_mfw = 5000

normalization_dict, dict_lower = word_translate_table_to_dict(normalization_table_path)

print(normalization_dict)
print(dict_lower)

normalization_list = ["tfidf"]

for normalization in normalization_list:
    # for lemmatization == True
    raw_corpus_object = DTM(corpus_path=corpus_path, dehyphen=True, lemmatize=True,
                            translate_umlaute=False, inverse_translate_umlaute=False,
                            sz_to_ss=False,
                            normalize_orthogr=True, normalization_table_path=normalization_table_path,
                            correct_ocr=True, eliminiate_pagecounts=True, n_mfw=n_mfw,
                            eliminate_pos_items=True,

                           normalize=normalization, language_model=my_model_de_path)
    outfile_basename = "raw_dtm_lemmatized_"+str(normalization)+"_"+str(n_mfw)+"mfw.csv"
    raw_dtm_outfile_path = os.path.join(outfile_dtm_path, outfile_basename)
    raw_corpus_object.generate_from_textcorpus()
    raw_corpus_object.save_csv(raw_dtm_outfile_path)

    # for lemmatization == False
    #raw_corpus_object = DTM(corpus_path=corpus_path, dehyphen=True, lemmatize=False, correct_ocr=True, eliminiate_pagecounts=True, inverse_translate_umlaute=False, n_mfw=n_mfw,
    #                               normalize=normalization)
    #outfile_basename = "raw_dtm_5000mfw_non-lemmatized_" + str(normalization) + "_" + str(n_mfw) + "mfw.csv"
    #raw_dtm_outfile_path = os.path.join(outfile_dtm_path, outfile_basename)
    #raw_corpus_object.generate_from_textcorpus()
    #raw_corpus_object.save_csv(raw_dtm_outfile_path)


