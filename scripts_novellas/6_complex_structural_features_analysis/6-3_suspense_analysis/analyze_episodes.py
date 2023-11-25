system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/PyNovellaHistory')


from preprocessing.prepare_corpus import select_all_episodes
from preprocessing.presetting import global_corpus_directory, language_model_path, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix, generate_text_files
import os

language_model = language_model_path(system)

corpus_path ="/mnt/data/users/schroeter/novella_corpus_with_episodes/07_Projektdaten_txt_gesamt_mit_ID"
outfile_path = os.path.join(local_temp_directory(system), "novella_episodes")
episodes_list = select_all_episodes(corpus_path)
print(episodes_list)

generate_text_files(chunking=False, pos_representation=False, corpus_path=corpus_path, outfile_path=outfile_path,
                        only_selected_files=True, list_of_file_ids=episodes_list, language_model=language_model,
                        correct_ocr=True, eliminate_pagecounts=True,
                 handle_special_characters=True, inverse_translate_umlaute=False, lemmatize=False,
                 remove_hyphen=True, sz_to_ss=False, translate_umlaute=False,
                 eliminate_pos_items=False, list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"],
                 keep_pos_items=False, list_keep_pos_tags=None,
                 segmentation_type="fixed", fixed_chunk_length=600, num_chunks=5,
                        normalize_orthogr=False, normalization_table_path=None,
                        reduce_to_words_from_list=False, reduction_word_list=None,
                 stopword_list=None, remove_stopwords=None)

columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"embedding_Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe", "Brustwarzen": "Erotik"}



