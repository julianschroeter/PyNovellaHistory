system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')

import os
import pandas as pd
import numpy as np

from preprocessing.presetting import local_temp_directory,conll_base_directory, load_stoplist, vocab_lists_dicts_directory, word_translate_table_to_dict, global_corpus_representation_directory
from preprocessing.corpus_alt import DTM


metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

meta_df = pd.read_csv(metadata_filepath, index_col=0)
print(meta_df)

#heftroman_stoplist = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "heftroman_stopwords_header.txt"))

corpus_path = conll_base_directory(system)
outfile_directory = os.path.join(local_temp_directory(system), "conll_chunks_novellas")

if not os.path.exists(outfile_directory):
    os.makedirs(outfile_directory)

corpus_names_path = os.path.join(vocab_lists_dicts_directory(system), "lists_prenames", "rev_names_edjs.txt")
corpus_names_list = load_stoplist(corpus_names_path)
for filename in os.listdir(corpus_path):

    if ".tsv" in filename:

        filepath = os.path.join(corpus_path, filename)
        df = pd.read_csv(filepath,engine="python" , sep="\t", on_bad_lines="skip",names= ["Token", "Space_after_bool",
                                                                                          "regular/symbol", "sentence_count",
                                                                                          "nr_token_in_sent",
                                                                                          "filepath", "POS_SoMeWeTa",
                                                                                          "POSS_RNNT", "RNNT_prob",
                                                                                          "POS_Parzu","UPOS_SoMe",
                                                                                          "UPOS_RNNT", "UPOS_RNNT_prob",
                                                                                          "Morph_RNNT", "Morph_RNNTprob",
                                                                                          "Morph_Parzu", "Lemma_RNNL", "RNNL_prob",
                                                                                          "Lemma_Parzu", "head_Parzu",
                                                                                          "DepRel_Parzu", "Direkte_Rede_Bool",
                                                                                          "Direkte_Rede_prob", "Indirekte_Rede_bool",
                                                                                          "Indirekte_Rede_prob",
                                                                                          "Rep_Speech_Bool", "Rep_Speech_prob",
                                                                                          "FID_Bool", "FID_prob", "Coref_Cluster",
                                                                                          "NER_FLERT", "NER_Scores", "NER_ID"])
        df = df.iloc[1:, :] # drop first row with old but inconsistent column names from csv file
        dfs = np.array_split(df, 5)
        for i, chunk_df in enumerate(dfs):
            print(chunk_df)
            basename = os.path.basename(filepath)
            basename = os.path.splitext(basename)[0]
            basename = basename.replace(".txt", "")
            basename = basename.replace(".tsv", "")
            chunk_filename = "{}{}{:04d}".format(basename, "_", i)
            chunk_df.to_csv(os.path.join(outfile_directory, chunk_filename))

print("process finished")