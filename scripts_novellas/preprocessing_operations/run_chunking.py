system = "wcph113" # "my_mac" # "wcph104"

if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

from preprocessing.corpus import ChunkCorpus
from preprocessing.presetting import global_corpus_directory, global_corpus_representation_directory , load_stoplist, local_temp_directory, language_model_path, vocab_lists_dicts_directory
import os



corpus_path = global_corpus_directory(system, test=False)
chunks5parts_directory = os.path.join(local_temp_directory(system), "chunks5parts")
chunks_fixed_directory = os.path.join(local_temp_directory(system), "chunks_fixed")
stopfilepath = os.path.join(vocab_lists_dicts_directory(system), "stopwords_all.txt")
stoplist = load_stoplist(stopfilepath)
my_model_de = language_model_path(system)
normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")

junk_fixed_corpus_object = ChunkCorpus(corpus_path=corpus_path, outfile_directory=chunks_fixed_directory,
                                       correct_ocr=True, eliminate_pagecounts=True,
                                       handle_special_characters=True, inverse_translate_umlaute=False, lemmatize=True,
                                       normalize_orthogr=True, normalization_table_path=normalization_table_path,
                                       remove_hyphen=True, sz_to_ss=False, translate_umlaute=False, segmentation_type="fixed",
                                       fixed_chunk_length=1000, num_chunks=5,
                                       eliminate_pos_items=False, keep_pos_items=True,
                                       list_keep_pos_tags=["ADJ", "VERB", "NOUN", "ADV"],
                                       stopword_list=stoplist, remove_stopwords="after_chunking", language_model=my_model_de)


junk5parts_corpus_object = ChunkCorpus(corpus_path=corpus_path, outfile_directory=chunks5parts_directory,
                                       correct_ocr=True, eliminate_pagecounts=True,
                                       handle_special_characters=True,
                                       inverse_translate_umlaute=True, lemmatize=True,
                                       remove_hyphen=True, sz_to_ss=False, translate_umlaute=False,
                                       stoplist_filepath=None, segmentation_type="relative", fixed_chunk_length=600, num_chunks=5,
                                       eliminate_pos_items=True,
                                       list_of_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"])

junk_fixed_corpus_object()