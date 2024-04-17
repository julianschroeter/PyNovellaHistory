system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import random
import pandas as pd
from preprocessing.deeplearning import deploy_tokenizer
from preprocessing.presetting import local_temp_directory, global_corpus_representation_directory
from preprocessing.sampling import select_from_corpus_df
import pickle as pkl
import numpy as np
import os

# load data
sentences_df = pd.read_csv(os.path.join(global_corpus_representation_directory(system), "novella_corpus_as_sentences_for_BERT-class.csv"), index_col=0)

metadata_df = pd.read_csv(os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv"), index_col=0)

sentences_df["label"] = sentences_df.apply(lambda x: metadata_df.loc[x.doc_id,"Gattungslabel_ED_normalisiert"], axis=1)
df = sentences_df
df = df[df.isin({"label": ["R", "M"]}).any(1)]
print(df)
df.loc[:,"label"] = df.loc[:,"label"].replace({"R":1, "M":0 })
df_N = df[df["label"]== 1]
df_E = df[df["label"] == 0]

N_ids = list(set(df_N["doc_id"].values.tolist()))
E_ids = list(set(df_E["doc_id"].values.tolist()))

train_N_ids = random.sample(N_ids, int(0.7 * len(N_ids)))
test_N_ids = [x for x in N_ids if x not in train_N_ids]

train_E_ids = random.sample(E_ids, int(0.7 * len(E_ids)))
test_E_ids = [x for x in E_ids if x not in train_E_ids]

print(train_N_ids)
print(test_N_ids)

df = select_from_corpus_df(df,20)
df["label"] = df.apply(lambda x: metadata_df.loc[x.doc_id,"Gattungslabel_ED_normalisiert"], axis=1)
df.loc[:,"label"] = df.loc[:,"label"].replace({"R":1, "M":0 })
#df_N = df[df["label"]== 1]
#df_E = df[df["label"] == 0]


data = df

print(data)
# Iteration über Dataframe mit den Spalten "text" und "label"
X = []
Y = []
M = []
for index, row in data.iterrows():
    token, mask = deploy_tokenizer(row["sent_string"])
    print(token)
    print(mask)

    Y.append(int(row["label"]))
    X.append(token)
    M.append(mask)

with open(os.path.join(local_temp_directory(system), "Y.pkl"), "wb") as f:
    pkl.dump(np.array(Y), f)
with open(os.path.join(local_temp_directory(system),"X.pkl"), "wb") as f:
    pkl.dump(np.array(X), f)
with open(os.path.join(local_temp_directory(system), "M.pkl"), "wb") as f:
    pkl.dump(np.array(M), f)


print("finished!")