import os
import re
from preprocessing.text import Text
from preprocessing.presetting import global_corpus_directory, language_model_path, local_temp_directory
system = "wcph113"

corpus_path = global_corpus_directory(system)
my_model_de_path = language_model_path(system)
output_directory = os.path.join(local_temp_directory(system), "episodes")

def put_episodes_together(corpus_path, output_directory, correct_ocr=True, eliminate_pagecounts=False, handle_special_characters=False,
                          inverse_translate_umlaute=False, lemmatize=False, remove_hyphen=False, sz_to_ss=False, translate_umlaute=False,
                          eliminate_pos_items=False, list_eliminate_pos_tags=None, keep_pos_items=False, list_keep_pos_tags=None, remove_stopwords=False,
                          stopword_list=None, language_model= my_model_de_path):
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
                            lemmatize=lemmatize, remove_hyphen=remove_hyphen, sz_to_ss=sz_to_ss,
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



put_episodes_together(corpus_path, output_directory)