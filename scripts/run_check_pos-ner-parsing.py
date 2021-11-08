import spacy
from preprocessing.presetting import global_corpus_directory, set_DistReading_directory, local_temp_directory
from preprocessing.text import Text
import os
from preprocessing.corpus import check_save_pos_ner_parsing_corpus
system = "my_mac"
list_of_file_ids_Johannes = ["00104-00", "00117-00", "00119-00", "00136-00", "00138-00", "00139-00", "00162-00", "00190-00", "00191-00", "00195-00","00595-00","00597-00","00598-00"

,"00606-00"

,"00607-00"

,"00608-00"

,"00609-00"

,"00610-00"
]
list_of_file_ids_Theresa = ["00098-00",

"00130-00",

"00132-00",

"00141-00",

"00143-00",

"00181-00",

"00147-00",

"00121-00",

"00155-00",

"00177-00"]
my_model_de = "/Users/karolineschroter/Documents/CLS/Sprachmodelle"

corpus_path = global_corpus_directory(system_name=system)
outfile_path = chunks5parts_directory = os.path.join(local_temp_directory(system), "ner_test")
check_save_pos_ner_parsing_corpus(corpus_path, outfile_path, list_of_file_ids_Theresa, correct_ocr=True, eliminate_pagecounts=True,
                 handle_special_characters=True, inverse_translate_umlaute=False, lemmatize=True,
                 remove_hyphen=True, sz_to_ss=False, translate_umlaute=False,
                 eliminate_pos_items=True, list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"],
                 keep_pos_items=False, list_keep_pos_tags=None,
                 segmentation_type=None, fixed_chunk_length=None, num_chunks=None,
                 stopword_list=None, remove_stopwords=None, language_model=my_model_de)



