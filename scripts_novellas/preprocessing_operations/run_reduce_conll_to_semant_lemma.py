system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')

import os
import pandas as pd
import numpy as np

from preprocessing.presetting import local_temp_directory, load_stoplist, vocab_lists_dicts_directory, global_corpus_representation_directory
from preprocessing.corpus_alt import DTM


metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


corpus_path = os.path.join(local_temp_directory(system), "conll_chunks_novellas")

outfile_dir_names = os.path.join(local_temp_directory(system), "Namen_novellas_chunks")
outfile_dir_semant = os.path.join(local_temp_directory(system), "SemantLemma_novellas_chunks")

if not os.path.exists(outfile_dir_names):
    os.makedirs(outfile_dir_names)
if not os.path.exists(outfile_dir_semant):
    os.makedirs(outfile_dir_semant)

corpus_names_path = os.path.join(vocab_lists_dicts_directory(system), "lists_prenames", "rev_names_edjs.txt")
corpus_names_list = load_stoplist(corpus_names_path)
for filename in os.listdir(corpus_path):
    doc_basename = filename[:-5]
    print(doc_basename)

    if doc_basename == doc_basename:
        filepath = os.path.join(corpus_path, filename)
        df = pd.read_csv(filepath,engine="python" , sep=",", on_bad_lines="skip")


        #df["Coref"] = df["Coref"].astype("category")
        df_per = df[df["NER_FLERT"] == "['PER']"]

        grouped = df_per.groupby(by= "NER_ID")
        per_dfs = [grouped.get_group(df) for df in grouped.groups]
        id_name_dic = {}
        for df in per_dfs:

            most_freq_name = df["Lemma_RNNL"].value_counts().idxmax()

            id = df["NER_ID"].value_counts().idxmax()
            id_name_dic[id] = most_freq_name
        print(id_name_dic)

        df_per["NE_name"] = df_per["NER_ID"].apply(lambda x: id_name_dic[x] )
        print(df_per)




        pos_dict = {"UPOS_RNNT": ["NOUN", "VERB", "ADJ", "ADV"]}
        semant_df = df[df.isin(pos_dict).any(1)]
        names_df = df_per

        semant_lemma_list = semant_df["Lemma_RNNL"].values.tolist()
        #semant_lemma_list = [str(word) for word in semant_lemma_list if word not in heftroman_stoplist]

        names_list = names_df["NE_name"].values.tolist()
        #names_list = [str(word) for word in names_list if word not in heftroman_stoplist]
        names_list = [word for word in names_list if word in corpus_names_list]

        semant_text = " ".join(map(str, semant_lemma_list))
        names_text = " ".join(map(str, names_list))

        outfilepath = os.path.join(outfile_dir_names, filename)
        with open(outfilepath, "w") as f:
            f.write(names_text)

        outfilepath = os.path.join(outfile_dir_semant, filename)
        with open(outfilepath, "w") as f:
            f.write(semant_text)

    print("file with filename ", filename, "successfully processed")

print("process finished")

