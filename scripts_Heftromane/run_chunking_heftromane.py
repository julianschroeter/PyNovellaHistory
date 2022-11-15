system = "wcph113" # "my_mac" # "wcph104"

if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

from preprocessing.corpus_alt import ChunkCorpus
from preprocessing.presetting import global_corpus_directory, global_corpus_representation_directory , load_stoplist, local_temp_directory, language_model_path, vocab_lists_dicts_directory
import os



corpus_path = os.path.join(local_temp_directory(system), "heftromane_chunking_start")

chunks_fixed_directory = os.path.join(local_temp_directory(system), "heftromane_chunks_fixed_fullnames")
stopfilepath = os.path.join(vocab_lists_dicts_directory(system), "stopwords_all.txt")
stoplist = load_stoplist(stopfilepath)
my_model_de = language_model_path(system)
normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")

junk_fixed_corpus_object = ChunkCorpus(corpus_path=corpus_path, outfile_directory=chunks_fixed_directory,
                                       correct_ocr=False, eliminate_pagecounts=False,
                                       handle_special_characters=False, inverse_translate_umlaute=False, lemmatize=False,
                                       normalize_orthogr=False, normalization_table_path=False,
                                       remove_hyphen=False, sz_to_ss=False, translate_umlaute=False, segmentation_type="fixed",
                                       fixed_chunk_length=1000, num_chunks=5,
                                       eliminate_pos_items=False, keep_pos_items=False,
                                       list_keep_pos_tags=["ADJ", "VERB", "NOUN", "ADV"],
                                       stopword_list=stoplist, remove_stopwords=False, language_model=my_model_de)


junk_fixed_corpus_object()