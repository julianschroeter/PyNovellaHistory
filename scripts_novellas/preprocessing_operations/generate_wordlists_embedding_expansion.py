# Imports
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, save_stoplist, keywords_to_semantic_fields, load_stoplist
import os

system = "my_xps" # "wcph113"

my_language_model_de = "de_core_news_lg"


lists_path = vocab_lists_dicts_directory(system)
for filename in os.listdir(lists_path):
    print(filename)
    if "Ausgangsliste_ploetzlich" in filename:
        input_words = load_stoplist(os.path.join(lists_path, filename))
        print(input_words)
        res_filename = filename.replace("Ausgangsliste", "ResultList")
        semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=50,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

        input_words.extend(semantic_words_list)
        new_list = list(set(input_words))
        print(new_list)
        wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), res_filename)
        save_stoplist(new_list, wordlist_filepath)