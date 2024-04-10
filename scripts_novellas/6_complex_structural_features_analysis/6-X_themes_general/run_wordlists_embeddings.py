# Imports
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, save_stoplist, keywords_to_semantic_fields
import os

system = "wcph113"

my_language_model_de = language_model_path(system)


input_words = ["Dorf", "Hof", "Knecht", "Feld", "Wald", "Förster", "Pflug", "Bauernhof", "Stall", "Pferd", "Kuh", "Landwirtschaft", "Bauer",
               "melken", "Acker", "Dorfkirche", "Dorfplatz", "bäuerlich", "Landwirt"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Italien", "Frankreich", "Spanien", "italienisch", "französisch", "spanisch",
               "Nordfrankreich", "Südfrankreich", "Provence", "Paris", "Lyon", "Marseille", "Bordeaux",
                "Neapel", "Sizilien", "Toscana", "Florenz", "Venedig", "Rom", "Genua", "Madrid", "Salamanca"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
romaniawordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), "romania_wordlist.txt")
save_stoplist(semantic_words_list, romaniawordlist_filepath)


input_words = ["Gespenst", "Geister", "Geisterwesen", "Geistwesen", "Schlossgespenst", "Schauder", "Schauer", "Entsetzen",
               "unheimlich", "ungeheuer", "ungeheuerlich", "wahnsinnig", "furchtbar", "zittern"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), "grusel_wordlist.txt")
save_stoplist(semantic_words_list, wordlist_filepath)



input_words = ["König", "Königin", "Prinz", "Prinzessin", "Graf", "Gräfin", "Kaiser", "Kaiserin",
               "Fürst", "Fürstin", "Untertan", "Herrscher", "herrschen", "Baron", "Baronin"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), "herrschertitel_wordlist.txt")
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Liebe", "lieben", "verliebt", "Zuneigung", "Verehrung", "verehren"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), "Liebe_wordlist.txt")
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Verzweiflung", "Tod", "Mord", "Verderben", "töten", "tot"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), "Verzweiflung_wordlist.txt")
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Waffe", "Messer", "Dolch", "Degen", "Schwert", "Blut", "Verletzung", "Gewehr", "Pistole", "Schuss", "Kugel"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Verstand", "Vernunft", "Geist", "denken", "nachdenken", "Gedanke", "Idee", "Intelligenz", "Rationalität", "geistreich", "geistvoll"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Großmutter", "Großvater", "Kind", "Enkel", "Enkelin", "Mutter", "Vater", "Mama", "Papa", "Tochter", "Sohn"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Offizier", "Soldat", "General", "Heer", "Armee", "Krieg", "Militär", "Oberst", "Kamerad", "Rittmeister", "Säbel", "Truppe"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)


input_words = ["Meer", "Schiff", "Wind", "Galeere", "Segel", "Welle", "Sturm", "Mast", "Ufer", "Bord", "Insel", "Kahn", "Matrose", "Küste"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Wollust", "küssen", "Schenkel", "Amor", "Genuss", "Erotik", "Sex", "Bett"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Theater", "Bühne", "Schauspieler", "Vorhang", "Publikum", "Zuschauer", "Beifall", "Loge", "Szene", "Spiel", "Komödie", "Aufführung", "Künstler", "Direktor"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Novelle", "erzählen", "gesellig", "Geselligkeit", "Gemeinschaft", "Erzählung"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Theater", "Bühne", "Schauspieler", "Vorhang", "Publikum", "Zuschauer", "Beifall", "Loge", "Szene", "Spiel", "Komödie", "Aufführung", "Künstler", "Direktor"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Regen", "Blitz", "Donner", "Sturm", "Orkan", "Gewitter", "Unwetter", "regnen", "donnern", "hageln", "Donnerschlag"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["Hochzeit", "Heirat", "Braut", "Bräutigam", "Brautvater", "Trauung", "Glück", "Ehe", "Eheglück"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

input_words = ["töten", "verletzen", "stechen", "zustechen", "schneiden", "stoßen", "erschießen", "schießen", "verwunden", "morden", "ermorden"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
print(semantic_words_list)
filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

