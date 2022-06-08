
def calculate_share(text_as_list, ne_list, normalize='l1', standardize=False, case_sensitive=False):
    """
    proceeds the calculation operation and stores the result wordlist_shares attribute.
    """
    # text_as_list = text.split(" ")


    hits = 0
    if case_sensitive == True:
        for token in text_as_list:
            if token in ne_list:
                hits += 1
    elif case_sensitive == False:
        wordlist_lower = [token.lower() for token in ne_list]
        for token in text_as_list:
            if token.lower() in wordlist_lower:
                hits += 1

    if normalize ==  "abs" and standardize == False:
        share = hits
    elif normalize == "l1" and standardize == False:
        share = hits / len(text_as_list)
    elif normalize == "l1" and standardize == True:
        share = hits / (len(text_as_list) * len(ne_list))


    return share


import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from preprocessing.presetting import global_corpus_representation_directory, vocab_lists_dicts_directory, load_stoplist, merge_several_stopfiles_to_list

system = "my_xps" # "wcph113"

list_path = os.path.join(vocab_lists_dicts_directory(system), "toponym_lists", "italian.txt")
italian_list = load_stoplist(list_path)

list_path = os.path.join(vocab_lists_dicts_directory(system), "toponym_lists", "french.txt")
french_list = load_stoplist(list_path)

list_path = os.path.join(vocab_lists_dicts_directory(system), "toponym_lists", "spanish.txt")
spanish_list = load_stoplist(list_path)


rom_list = []
for top_list in [italian_list, french_list, spanish_list]:
    for item in top_list:
        rom_list.append(item)
print(rom_list)
rom_list = list(set(rom_list))


df_infilepath = os.path.join(global_corpus_representation_directory(system), "NEs_document_Matrix_test.csv")
df = pd.read_csv(df_infilepath, index_col=0)

df_outfilepath = os.path.join(global_corpus_representation_directory(system), "toponym_share_Matrix.csv")

print(df)

df["it_top"] = df["Orte"].apply(lambda x: calculate_share(str(x).split(". "), ne_list=italian_list))
df["fr_top"] = df["Orte"].apply(lambda x: calculate_share(str(x).split(". "), ne_list=french_list))
df["sp_top"] = df["Orte"].apply(lambda x: calculate_share(str(x).split(". "), ne_list=spanish_list))
df["rom_top"] = df["Orte"].apply(lambda  x: calculate_share(str(x).split(". "), ne_list=rom_list))

df = (df[["it_top", "fr_top", "sp_top", "rom_top"]])

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df.to_numpy()), columns=df.columns, index=df.index)

print(df)

df.to_csv(path_or_buf=df_outfilepath)