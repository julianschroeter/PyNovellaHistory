system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')

import os
import re
import pandas as pd
import numpy as np

from preprocessing.presetting import local_temp_directory, load_stoplist, vocab_lists_dicts_directory, global_corpus_representation_directory
from preprocessing.corpus import DTM


metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


corpus_path = os.path.join(local_temp_directory(system), "conll_novellas")
print(corpus_path)

length_dic = {}
for filename in os.listdir(corpus_path):
    if re.search("\d{5}-\d{2}", filename):
        text_id =  re.search("\d{5}-\d{2}", filename).group()
        filepath = os.path.join(corpus_path, filename)
        df = pd.read_csv(filepath, engine="python", sep=",", on_bad_lines="skip")
        df = df[df["regular/symbol"] == "regular"]
        sent_count = df.iloc[-1,4]
        token_count = df.shape[0]
        print(sent_count)
        length_dic[text_id] = [token_count, sent_count]


length_df = pd.DataFrame.from_dict(length_dic,columns=["token_count", "sentence_count"],  orient="index")
print(length_df)


length_df.to_csv(os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv"))

