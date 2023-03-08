import os
import re
from preprocessing.text import Text



def join_episodes(corpus_path, output_directory, correct_ocr=True, eliminate_pagecounts=False, handle_special_characters=False,
                  inverse_translate_umlaute=False, lemmatize=False, remove_hyphen=False, normalize_orthogr=False, normalization_table_path=None,
                  sz_to_ss=False, translate_umlaute=False,
                  eliminate_pos_items=False, list_eliminate_pos_tags=None, keep_pos_items=False, list_keep_pos_tags=None, remove_stopwords=False,
                  stopword_list=None, language_model=None):
    """
    puts all files with the separated episodes in one txt file. episodes are identified by the document id XXXXX-01, XXXXX-02, XXXXX-03 and so on.
    Based on Text class, manipulations such as correction, lemmatization and so on, can be performed on the text
    returns a text file with XXXXX-00.txt as filename
    """

    first_episodes_full_ids = []
    full_ids_list = []
    for filename in os.listdir(corpus_path):
        full_id = re.search("\d{5}-\d{2}", filename).group() if re.search("\d{5}-\d{2}", filename) else None
        if full_id:
            full_ids_list.append(full_id)
            work_id = full_id[:5]
            episode_id = full_id[6:8]
            episode_count = 1
            if episode_id == "{:02}".format(episode_count):
                first_episodes_full_ids.append(full_id)

    for id in first_episodes_full_ids:
        text = ""
        work_id = id[:5]
        episode_count = 1
        next_episode_string = "{:02}".format(episode_count)
        expected_next_episode_full_id = str(work_id + "-" + next_episode_string)
        while expected_next_episode_full_id in full_ids_list:
            for filename in os.listdir(corpus_path):
                text_obj = Text(filepath=os.path.join(corpus_path, filename), correct_ocr=correct_ocr,
                            eliminate_pagecounts=eliminate_pagecounts,
                            handle_special_characters=handle_special_characters,
                            inverse_translate_umlaute=inverse_translate_umlaute,
                            lemmatize=lemmatize, remove_hyphen=remove_hyphen, normalize_orthogr=normalize_orthogr, normalization_table_path=normalization_table_path,
                                sz_to_ss=sz_to_ss,
                            translate_umlaute=translate_umlaute,
                            eliminate_pos_items=eliminate_pos_items, list_eliminate_pos_tags=list_eliminate_pos_tags,
                            keep_pos_items=keep_pos_items, list_keep_pos_tags=list_keep_pos_tags,
                            remove_stopwords=remove_stopwords, stopword_list=stopword_list,
                            language_model=language_model,
                            )
                text_obj.f_extract_id()

                if text_obj.id == expected_next_episode_full_id:
                    text_obj()
                    text = str(text + text_obj.text + "\n")
            episode_count += 1
            next_episode_string = "{:02}".format(episode_count)
            expected_next_episode_full_id = str(work_id + "-" + next_episode_string)

        filename = str(work_id + "-00" + ".txt")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        with open(os.path.join(output_directory, filename), "w", encoding="utf8") as file:
            file.write(text)


def join_all_documents(corpus_path, outfile_directory, outfile_name, correct_ocr=True, eliminate_pagecounts=False, handle_special_characters=False,
                  inverse_translate_umlaute=False, lemmatize=False, remove_hyphen=False,
                       normalize_orthogr=False, normalization_table_path=None,
                       sz_to_ss=False, translate_umlaute=False,
                  eliminate_pos_items=False, list_eliminate_pos_tags=None, keep_pos_items=False, list_keep_pos_tags=None, remove_stopwords=False,
                       reduce_to_words_from_list =False, reduction_word_list=None,
                  stopword_list=None, language_model=None):
    text = ""
    for filename in os.listdir(corpus_path):
        print("currently processes file: ", filename)
        text_obj = Text(filepath=os.path.join(corpus_path, filename), correct_ocr=correct_ocr,
                        eliminate_pagecounts=eliminate_pagecounts,
                        handle_special_characters=handle_special_characters,
                        inverse_translate_umlaute=inverse_translate_umlaute,
                        normalize_orthogr=normalize_orthogr, normalization_table_path=normalization_table_path,
                        lemmatize=lemmatize, reduce_to_words_from_list=reduce_to_words_from_list, reduction_word_list=reduction_word_list,
                        remove_hyphen=remove_hyphen, sz_to_ss=sz_to_ss,
                        translate_umlaute=translate_umlaute,
                        eliminate_pos_items=eliminate_pos_items, list_eliminate_pos_tags=list_eliminate_pos_tags,
                        keep_pos_items=keep_pos_items, list_keep_pos_tags=list_keep_pos_tags,
                        remove_stopwords=remove_stopwords, stopword_list=stopword_list,
                        language_model=language_model,
                        )
        text_obj.f_extract_id()
        text_obj()
        print(text_obj.text)
        text = str(text + text_obj.text + "\n")

        if not os.path.exists(outfile_directory):
            os.makedirs(outfile_directory)
        with open(os.path.join(outfile_directory, outfile_name), "w", encoding="utf8") as file:
            file.write(text)


def select_all_episodes(corpus_path):
    """
    RETURN all filename ids for all files in corpus path, which are episodes, as a list
    """
    list_of_fileIDs = []
    for filename in os.listdir(corpus_path):
        full_id = re.search("\d{5}-\d{2}", filename).group() if re.search("\d{5}-\d{2}", filename) else None
        if full_id:
            episode_id = full_id[6:8]
            if episode_id != "00":
                list_of_fileIDs.append(full_id)
    return list_of_fileIDs    


