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


df_start = df.copy()

df = df_start[df_start["year"] < 1850]

columns = df.columns.values.tolist()
print(columns)
variables = [ "plötzlich", "suspense_sum"] # 'Gewaltverbrechen', 'Kampf', 'Entführung', 'Krieg', "Spuk", 'Gefahrenlevel', 'embedding_Angstempfinden', 'Angstempfinden', 'UnbekannteEindr', 'Feuer',

fig, axes = plt.subplots(2,2)
i = 0
for variable in variables:

    # for all chunks with value > 0
    new_df = df[df[variable] != 0]
    print(new_df)
    new_df.boxplot(by="Textabschnitt", column=variable, ax=axes[i,0])
    #means = df.groupby("Textabschnitt").median()
    #print(means)
    #plt.plot(means.index.values.tolist(), means[variable])
    axes[i,0].set_xlabel("Vor 1850")
    axes[i,0].set_ylim(0,1)
    if variable == "suspense_sum":
        axes[i,0].set_title("Alle Gefahrentypen")
    elif variable == "plötzlich":
        axes[i, 0].set_title("Wort: plötzlich")
    i +=1


df = df_start[df_start["year"] >= 1850]

columns = df.columns.values.tolist()
print(columns)
variables = ["plötzlich",
             "suspense_sum"]  # 'Gewaltverbrechen', 'Kampf', 'Entführung', 'Krieg', "Spuk", 'Gefahrenlevel', 'embedding_Angstempfinden', 'Angstempfinden', 'UnbekannteEindr', 'Feuer',


i = 0
for variable in variables:
    # for all chunks with value > 0
    new_df = df[df[variable] != 0]
    print(new_df)
    axes[i,1] = new_df.boxplot(by="Textabschnitt", column=variable, ax=axes[i,1])
    # means = df.groupby("Textabschnitt").median()
    # print(means)
    # plt.plot(means.index.values.tolist(), means[variable])
    axes[i,1].set_xlabel("Nach 1850")
    axes[i,1].set_ylim(0, 1)
    if variable == "suspense_sum":
        axes[i, 1].set_title("Alle Gefahrentypen")
    elif variable == "plötzlich":
        axes[i, 1].set_title("Wort: plötzlich")
    i += 1
fig.suptitle("Dominanz von Spannungsindikatoren in den Teilabschnitten jeder Episode")
fig.supxlabel("Nummer der Episode")
fig.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system), "figures", "Boxplots_ploetzlich_suspense_within_episodes.svg"))
plt.show()