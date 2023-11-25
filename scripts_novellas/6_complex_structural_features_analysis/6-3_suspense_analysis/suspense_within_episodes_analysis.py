system = 'my_xps' # "wcph113"

import pandas as pd
import os
import matplotlib.pyplot as plt

from preprocessing.presetting import local_temp_directory, global_corpus_representation_directory
from preprocessing.corpus import DocFeatureMatrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()

df = pd.read_csv(os.path.join(local_temp_directory(system), os.path.join(local_temp_directory(system), "All_Chunks_Danger_FearCharacters_episodes.csv")), index_col=0)
novellas_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
metadata_df = pd.read_csv(novellas_metadata_filepath, index_col=0)
print(df)

df["genre"] = df.apply(lambda x: metadata_df.loc[str(x.doc_id[:6]+"00"), "Gattungslabel_ED_normalisiert"], axis=1)
df["year"] = df.apply(lambda x: metadata_df.loc[str(x.doc_id[:6]+"00"), "Jahr_ED"], axis=1)

df = df.rename(columns={"max_value":"Gefahrenlevel", "chunk_id":"Textabschnitt"})
df["Textabschnitt"] = df["Textabschnitt"].astype(int)
df["Textabschnitt"] = df["Textabschnitt"].apply(lambda x: x + 1)

df["episode"] = df["doc_id"].apply(lambda x: x[-2:]).astype(int)
df["doc"] = df["doc_id"].apply(lambda x: x[:5])

df["suspense_sum"] = df.apply(lambda x: x.Gewaltverbrechen + x.Kampf + x.Entführung + x.Krieg + x.Spuk + x.Angstempfinden + x.UnbekannteEindr + x.Feuer + x.plötzlich, axis=1)
df["suspense_sum"] = scaler.fit_transform(df.apply(lambda x: x.Gewaltverbrechen + x.Kampf + x.Entführung , axis=1).values.reshape(-1,1))

# remove the final episode of each document
idx = df.groupby("doc")["episode"].transform(max) == df["episode"]
print(idx)
df = df.loc[df.index.difference(idx)]


df_all = df.copy()

df = df[df["year"] >= 1850]

columns = df.columns.values.tolist()
print(columns)
variables = ['Gewaltverbrechen', 'Kampf', 'Entführung', 'Krieg', "Spuk", 'Gefahrenlevel', 'embedding_Angstempfinden', 'Angstempfinden', 'UnbekannteEindr', 'Feuer', "plötzlich", "suspense_sum"]

for variable in variables:
    # for all chunks with value > 0
    new_df = df[df[variable] != 0]
    print(new_df)
    new_df.boxplot(by="Textabschnitt", column=variable)
    #means = df.groupby("Textabschnitt").median()
    #print(means)
    #plt.plot(means.index.values.tolist(), means[variable])
    plt.xlabel("Nach 1850")
    plt.ylim(0,1)
    plt.show()