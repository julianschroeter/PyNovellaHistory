import os
from preprocessing.presetting import language_model_path, save_stoplist, load_stoplist, vocab_lists_dicts_directory, global_corpus_representation_directory, merge_several_stopfiles_to_list
import spacy

system_name = "wcph113" #my_mac" # "wcph104"
language_model = language_model_path(system_name)


filepath1 = os.path.join(vocab_lists_dicts_directory(system_name), "de_stopwords_names.txt")
filepath2 = os.path.join(vocab_lists_dicts_directory(system_name), "stopwords_all.txt")
filepath3 = os.path.join(vocab_lists_dicts_directory(system_name), "ocr_fehler.txt")
filepath4 = os.path.join(vocab_lists_dicts_directory(system_name), "names.txt")
start_list = merge_several_stopfiles_to_list([filepath1, filepath2, filepath3, filepath4])

romania_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system_name), "romania_wordlist.txt"))
verben_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system_name), "verben_sprechen_handeln.txt"))
reduced_list = []


nlp = spacy.load(language_model)
doc = nlp(" ".join(start_list))
for token in doc:
    print(token.text, token.pos_, token.ent_type_)



de_liste = []
functionwords = []
names_list = []
names_list_PROPN = []
locations_list = []
rest_list = characters_list = []
other_NER_list = []


for token in doc:
    if token.pos_ in ["ADJ", "ADV", "SCONJ", "ADP", "AUX", "PART", "DET", "PRON", "INTJ", "PART", "VERB", "CCONJ"]:
        functionwords.append((token.text, token.pos_, token.ent_type_))
    else:
        de_liste.append(token.text)




    if token.ent_type_:
        if token.ent_type_ == "PER":
            names_list.append((token.text, token.pos_, token.ent_type_))
        elif token.ent_type_ == "LOC":
            locations_list.append(token.text)
    elif token.pos_ in ["PROPN", "VERB", "NOUN", "ADJ"]:
        names_list_PROPN.append(token.text)
    else:
        rest_list.append((token.text, token.pos_, token.ent_type_))


print("Deutsche Namensliste:", names_list)
print("Namen, PROPN:", names_list_PROPN)
print("dt. Liste der Ortsnamen:", locations_list)
print("Resteliste:", rest_list)

manual_red_loc_list = ['amsterdam', 'florenz', 'frankfurt', 'genf', 'granada', 'holland', 'mainz', 'neapel', 'paris', 'spanien', 'venedig', 'wien']
locations_list = manual_red_loc_list
locations_list.extend(romania_list)
locations_list_lower = [word.lower() for word in locations_list]

print("extended loc list: ", locations_list)
for word in start_list:
    if word not in locations_list_lower:
        reduced_list.append(word)

reduced_list.extend(verben_list)


reduced_list = sorted(set(reduced_list))
print("reduzierte Liste: ", reduced_list)

save_stoplist(reduced_list, outfilepath=os.path.join(vocab_lists_dicts_directory(system_name), "stoplist_without_locations.txt"))