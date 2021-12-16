# Imports
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, save_stoplist, keywords_to_semantic_fields
import os

system = "wcph113"

my_language_model_de = language_model_path(system)


input_words = ["Liebe", "lieben", "verliebt", "Leidenschaft", "rasend", "Raserei", "verrückt", "Verrücktheit", "Täuschung", "täuschen", "Eifersucht", "Ehebruch",
                 "besessen", "Besessenheit", "Betrug", "betrügen", "Spiel", "Spielerei", "Liebschaft"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

filename = str(str(semantic_words_list[0])+"_amourpassion_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)


input_words = ["Liebe", "lieben", "verliebt", "ideal", "schön", "Schönheit", "tadellos", "perfekt", "wunderschön"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

filename = str(str(semantic_words_list[0])+"_finamour_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Liebe", "lieben", "verliebt", "Ehe", "Partnerschaft", "Heirat", "Harmonie", "Ehestreit", "Vernunft", "Freundschaft", "Kameradschaft", "Begleiter", "Partner", "Tochter", "Sohn", "Kind", "Haushalt"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

filename = str(str(semantic_words_list[0])+"_ehe_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)


input_words = ["Liebe", "lieben", "verliebt", "einzigartig", "einmalig", "einzig"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

filename = str(str(semantic_words_list[0])+"_romantliebe_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)