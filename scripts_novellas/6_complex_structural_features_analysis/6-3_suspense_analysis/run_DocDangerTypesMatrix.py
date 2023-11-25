system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os

from preprocessing.presetting import vocab_lists_dicts_directory, load_stoplist, language_model_path, global_corpus_representation_directory, local_temp_directory
from semantic_analysis.themes import DocThemesMatrix

my_model_de = language_model_path(system)
print(my_model_de)



list_of_wordlists = []

for filepath in os.listdir(vocab_lists_dicts_directory(system)):
    if "ResultList" in filepath:
        print(filepath)
        wordlist = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), filepath))
        list_of_wordlists.append(wordlist)


print(list_of_wordlists)

normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")
corpus_folder = "episodes_5chunks" # "SemantLemma_novellas_episodes_chunks" #"ergaenzt_chunks5parts_semant_lemma" #
corpus_path = os.path.join(local_temp_directory(system), corpus_folder)
matrix_obj = DocThemesMatrix(list_of_wordlists=list_of_wordlists,
                             corpus_path= corpus_path,
                             remove_hyphen=False,
                             normalize_orthogr=False,
                             normalization_table_path=normalization_table_path,
                             correct_ocr=False,
                             eliminate_pagecounts=False,
                             handle_special_characters=False,
                             inverse_translate_umlaute=False,
                             keep_pos_items=False,
                             eliminate_pos_items=False,
                             list_of_pos_tags=None,
                             list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"],
                             lemmatize=False,
                             sz_to_ss=False,
                             translate_umlaute=False,
                             remove_stopwords=False,
                             language_model=my_model_de)
print(matrix_obj.data_matrix_df)

outfile_path = os.path.join(local_temp_directory(system), "DocDangerTypesMatrix_episodes.csv")
matrix_obj.save_csv(outfile_path)
