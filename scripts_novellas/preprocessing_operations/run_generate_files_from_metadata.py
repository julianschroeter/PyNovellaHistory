system = "wcph113" # "my_mac" "wcph104" "my_mac" "my_xps"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import pandas as pd
import os

from preprocessing.metadata_transformation import extract_file_ids_period, extract_ids_categorical
from preprocessing.presetting import language_model_path, global_corpus_representation_directory,vocab_lists_dicts_directory, global_corpus_directory, local_temp_directory
from preprocessing.corpus_alt import generate_text_files

my_model = language_model_path(system)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

input_directory = global_corpus_directory(system_name=system)
output_directory = os.path.join(local_temp_directory(system_name=system), "corpus_all_corrected")
normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")

# example: generate a list of file ids with metadate selection: (Jahr_ED before 1830)
metadata_df = pd.read_csv(metadata_filepath, index_col=0)
print(metadata_df["Jahr_ED"])
ids_list = extract_file_ids_period(df=metadata_df,from_year=0, to_year=1830)
print(ids_list)

nonnormalized_ids_list = extract_ids_categorical(df=metadata_df, metadata_category="Bearbeitungsqualität", variables_list=["original", "niedrig_original"])
print(nonnormalized_ids_list)

generate_text_files(outfile_path=output_directory, corpus_path=input_directory,
                    only_selected_files=True, list_of_file_ids=nonnormalized_ids_list,
                    chunking=False, lemmatize=False,
                    correct_ocr=True, eliminate_pagecounts=True,
                    remove_hyphen=True, handle_special_characters=True,
                    translate_umlaute=False, sz_to_ss=False, inverse_translate_umlaute=False,
                    eliminate_pos_items=False, keep_pos_items=False, remove_stopwords=False,
                     pos_representation=False,
                    language_model=my_model, normalize_orthogr=False,
                    normalization_table_path=normalization_table_path)

values_dict = {"Bearbeitungsqualität": ["original", "niedrig_original"]}
df = metadata_df
df = df[~df.isin(values_dict).any(1)]
normalized_ids = df.index.tolist()
print(normalized_ids)

generate_text_files(outfile_path=output_directory, corpus_path=input_directory,
                    only_selected_files=True, list_of_file_ids=normalized_ids,
                    chunking=False, lemmatize=False,
                    correct_ocr=True, eliminate_pagecounts=True,
                    remove_hyphen=False, handle_special_characters=True,
                    translate_umlaute=False, sz_to_ss=False, inverse_translate_umlaute=False,
                    eliminate_pos_items=False, keep_pos_items=False, remove_stopwords=False,
                     pos_representation=False,
                    language_model=my_model, normalize_orthogr=False,
                    normalization_table_path=normalization_table_path)