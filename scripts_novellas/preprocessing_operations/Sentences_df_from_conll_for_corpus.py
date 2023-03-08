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

sent_corpus_df = pd.DataFrame(columns=["doc_id", "sent_nr","sent_string", "sent_len"])

for filename in os.listdir(corpus_path):
    if re.search("\d{5}-\d{2}", filename):
        text_id =  re.search("\d{5}-\d{2}", filename).group()
        filepath = os.path.join(corpus_path, filename)
        df = pd.read_csv(filepath, engine="python", sep=",", on_bad_lines="skip")

        grouped = df.groupby(by="sentence_count")
        sent_dfs = [grouped.get_group(df) for df in grouped.groups]
        for sent_df in sent_dfs:

            words = sent_df["Token"].values.tolist()
            words = [str(word) for word in words]
            sent_str = re.sub(r"(?: ([.,;]))", r"\g<1>", " ".join(words))
            #sent_str = " ".join(sent_df["Token"].values.tolist())
            sent_nr = sent_df.iloc[0,4]

            print(sent_str)
            print(sent_nr)
            print(len(words))
            print(sent_df["nr_token_in_sent"].values.tolist()[-1])
            sent_corpus_df = sent_corpus_df.append({"doc_id":text_id, "sent_nr": sent_nr, "sent_string": sent_str, "sent_len": len(words)},
                                  ignore_index=True)

print(sent_corpus_df)


sent_corpus_df.to_csv(os.path.join(local_temp_directory(system), "novella_corpus_sentences.csv"))

