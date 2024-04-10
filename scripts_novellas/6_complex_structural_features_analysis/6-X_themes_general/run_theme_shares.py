system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os

from preprocessing.presetting import global_corpus_directory, local_temp_directory,vocab_lists_dicts_directory, load_stoplist, language_model_path, global_corpus_representation_directory
from semantic_analysis.themes import DocThemesMatrix

my_model_de = language_model_path(system)



list_of_wordlists = []

for filepath in os.listdir(vocab_lists_dicts_directory(system)):
    if "Ausgangsliste_plötzlich.txt" in filepath:
        wordlist = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), filepath))
        list_of_wordlists.append(wordlist)


print(list_of_wordlists)

normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")

corpus_path = os.path.join(local_temp_directory(system), "SemantLemma_novellas_episodes_chunks" )# global_corpus_directory(system)
matrix_obj = DocThemesMatrix(corpus_path= corpus_path, list_of_wordlists=list_of_wordlists,
                             list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"], list_of_pos_tags=None, keep_pos_items=False,
                             language_model=my_model_de, remove_hyphen=True, normalize_orthogr=True, normalization_table_path=normalization_table_path,
                             correct_ocr=True, eliminate_pagecounts=True, handle_special_characters=True,
                             inverse_translate_umlaute=False,
                             eliminate_pos_items=True,
                             lemmatize=True,
                             sz_to_ss=False, translate_umlaute=False,
                             remove_stopwords=False)
print(matrix_obj.data_matrix_df)

outfile_path = os.path.join(global_corpus_representation_directory(system), "ploetzlich_DocThemesMatrix_novellas_episodes.csv")
matrix_obj.save_csv(outfile_path)



