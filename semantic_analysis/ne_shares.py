from metrics.scores import calculate_share
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from preprocessing.presetting import global_corpus_representation_directory, vocab_lists_dicts_directory, load_stoplist

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


df_infilepath = os.path.join(global_corpus_representation_directory(system), "conll_locations_Matrix.csv")
#os.path.join(global_corpus_representation_directory(system), "conll_loc_share_Matrix.csv")
df = pd.read_csv(df_infilepath, index_col=0)

df_outfilepath = os.path.join(global_corpus_representation_directory(system), "conll_toponym_share_Matrix.csv")

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