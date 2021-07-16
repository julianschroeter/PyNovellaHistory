import os
from Preprocessing.Presetting import load_stoplist, global_corpus_representation_directory, merge_several_stopfiles_to_list
import spacy

system_name = "wcph104"

filepath1 = os.path.join(global_corpus_representation_directory(system_name), "de_stopwords_names.txt")
filepath2 = os.path.join(global_corpus_representation_directory(system_name), "stopwords_all.txt")
start_list = merge_several_stopfiles_to_list([filepath1, filepath2])


nlp_en = spacy.load("en_core_web_sm")

doc_en = nlp_en(" ".join(start_list))
for token in doc_en:
    print(token.text, token.pos_, token.ent_type_)

["NOUN", "PROPN"]
en_functionwords = []
de_liste = []
for token in doc_en:
    if token.pos_ in ["ADJ", "ADV", "SCONJ", "ADP", "AUX", "PART", "DET", "PRON", "INTJ", "PART", "VERB", "CCONJ"]:
        en_functionwords.append((token.text, token.pos_, token.ent_type_))
    else:
        de_liste.append(token.text)

print(en_functionwords)

nlp_de = spacy.load('de_core_news_sm')



doc = nlp_de(" ".join(de_liste))

names_list = []
names_list_PROPN = []
locations_list = []
rest_list = characters_list = []
other_NER_list = []

for token in doc:
    if token.ent_type_:
        if token.ent_type_ == "PER":
            names_list.append((token.text, token.pos_, token.ent_type_))
        elif token.ent_type_ in ["LOC", "ORG"] and token.pos_ == "PROPN":
            locations_list.append((token.text, token.pos_, token.ent_type_))
    elif token.pos_ in ["PROPN", "VERB", "NOUN", "ADJ"]:
        names_list_PROPN.append((token, token.pos_, token.ent_type_))
    else:
        rest_list.append((token.text, token.pos_, token.ent_type_))


print("Deutsche Namensliste:", names_list)
print("Namen, PROPN:", names_list_PROPN)
print("dt. Liste der Ortsnamen:", locations_list)
print("Resteliste:", rest_list)