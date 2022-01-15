system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os

from preprocessing.prepare_corpus import join_all_documents
from preprocessing.presetting import local_temp_directory, vocab_lists_dicts_directory, language_model_path, load_stoplist

corpus_path = os.path.join(local_temp_directory(system), "SunKoh_Heftromane_txt")
outfile_directory = local_temp_directory(system)
outfile_name = "joined_SunKoh_Heftromane.txt"
normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")
my_language_model_path = language_model_path(system)
list_keep_pos_tags = ["ADJ", "VERB", "NOUN", "ADV"]
reduction_list_path = os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt")
german_words_list = load_stoplist(reduction_list_path)

join_all_documents(corpus_path=corpus_path, outfile_directory=outfile_directory, outfile_name=outfile_name, correct_ocr=True, eliminate_pagecounts=True, handle_special_characters=True,
                  inverse_translate_umlaute=False, lemmatize=False, reduce_to_words_from_list=True, reduction_word_list=german_words_list,
                   remove_hyphen=True, normalize_orthogr=True, normalization_table_path=normalization_table_path,
                   sz_to_ss=False, translate_umlaute=False,
                  eliminate_pos_items=False, list_eliminate_pos_tags=None, keep_pos_items=True, list_keep_pos_tags=list_keep_pos_tags, remove_stopwords=False,
                  stopword_list=None, language_model=my_language_model_path)