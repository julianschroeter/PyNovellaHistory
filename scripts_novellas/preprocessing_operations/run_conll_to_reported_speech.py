system = "my_xps" #  "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os
import pandas as pd
import numpy as np

from preprocessing.presetting import local_temp_directory, load_stoplist, vocab_lists_dicts_directory, global_corpus_representation_directory
from preprocessing.corpus import DTM


metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


corpus_path = os.path.join(local_temp_directory(system), "conll_novellas")

#outfile_dir_names = os.path.join(local_temp_directory(system), "conll_Namen_novellas")
outfile_dir_speech = os.path.join(local_temp_directory(system), "conll_SemantLemma_speech")

if not os.path.exists(outfile_dir_speech):
    os.makedirs(outfile_dir_speech)
if not os.path.exists(outfile_dir_speech):
    os.makedirs(outfile_dir_speech)

result_dict = {}

corpus_names_path = os.path.join(vocab_lists_dicts_directory(system), "lists_prenames", "rev_names_edjs.txt")
corpus_names_list = load_stoplist(corpus_names_path)
for filename in os.listdir(corpus_path):
    if filename not in os.listdir(outfile_dir_speech):
        filepath = os.path.join(corpus_path, filename)
        print(filepath)
        df = pd.read_csv(filepath,engine="python" , sep=",", on_bad_lines="skip")

        df_dirspeech = df[df["Direkte_Rede_Bool"] == "yes"]
        df_indirspeech = df[df["Indirekte_Rede_bool"] == "yes"]
        df_repspeech = df[df["Rep_Speech_Bool"] == "yes"]
        df_fid = df[df["FID_Bool"] == "yes"]
        fraction_dirspeech = df_dirspeech.shape[0] / df.shape[0]
        fraction_indirspeech = df_indirspeech.shape[0] / df.shape[0]
        fraction_repspeech = df_repspeech.shape[0] / df.shape[0]
        fraction_fid = df_fid.shape[0] / df.shape[0]
        values = [fraction_dirspeech, fraction_indirspeech, fraction_repspeech, fraction_fid]
        print(values)
        result_dict[filename] = values


result_df = pd.DataFrame.from_dict(result_dict, orient="index", columns=["fraction_dirspeech", "fraction_indirspeech", "fraction_repspeech", "fraction_fid"])
result_df.to_csv(os.path.join(global_corpus_representation_directory(system), "speech_rep_Matrix.csv"))
