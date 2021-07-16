from Preprocessing.Corpus import Junk_Corpus
from Preprocessing.Presetting import global_corpus_directory, global_corpus_representation_directory , load_stoplist, local_temp_directory
import os

system = "my_mac" # "wcph104"

corpus_path = global_corpus_directory(system, test=False)
chunks5parts_directory = os.path.join(local_temp_directory(system), "chunks5parts")
chunks1000w_directory = os.path.join(local_temp_directory(system), "chunks1000w")
stopfilepath = os.path.join(local_temp_directory(system), "temp_stoplist.txt")
stoplist = load_stoplist(stopfilepath)
my_model_de = os.path.join("/Users/karolineschroter/Documents/CLS/Sprachmodelle", "my_model_de")

junk1000w_corpus_object = Junk_Corpus(corpus_path=corpus_path, outfile_directory=chunks1000w_directory,
                                      correct_ocr=True, eliminate_pagecounts=True,
                                 handle_special_characters=True, inverse_translate_umlaute=True, lemmatize=True,
                                 remove_hyphen=True, sz_to_ss=False, translate_umlaute=False, segmentation_type="fixed",
                                      fixed_chunk_length=1000, num_chunks=5,
                                 eliminate_pos_items=False, keep_pos_items=True,
                                      list_keep_pos_tags=["ADJ", "VERB", "NOUN", "ADV"],
                                      stopword_list=stoplist, remove_stopwords="after_chunking", language_model=my_model_de)


junk5parts_corpus_object = Junk_Corpus(corpus_path=corpus_path, outfile_directory=chunks5parts_directory,
                                 correct_ocr=True, eliminate_pagecounts=True,
                                 handle_special_characters=True,
                 inverse_translate_umlaute=True, lemmatize=True,
                                 remove_hyphen=True, sz_to_ss=False, translate_umlaute=False,
                 stoplist_filepath=None, segmentation_type="relative", fixed_chunk_length=600, num_chunks=5,
                                 eliminate_pos_items=True,
                                 list_of_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"])

junk1000w_corpus_object()