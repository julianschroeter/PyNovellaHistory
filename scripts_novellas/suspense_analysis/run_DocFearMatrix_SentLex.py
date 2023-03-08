system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')


from sent_analysis.fear import DocSentFearMatrix
from preprocessing.presetting import word_translate_table_to_dict, vocab_lists_dicts_directory, local_temp_directory, language_model_path

import os

my_model_de = language_model_path(system)
print(my_model_de)

normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")
fear_dict_path = os.path.join(vocab_lists_dicts_directory(system), "furcht.txt")

fear_dict = word_translate_table_to_dict(fear_dict_path, also_lower_case=False)

for k, v in fear_dict.items():
    fear_dict[k] = float(v)

print(fear_dict)

corpus_path = os.path.join(local_temp_directory(system), "SemantLemma_novellas_episodes_chunks")

matrix_obj = DocSentFearMatrix(sent_dict=fear_dict,
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

outfile_path = os.path.join(local_temp_directory(system), "DocSentFearMatrix_novellas_episodes.csv")
matrix_obj.save_csv(outfile_path)