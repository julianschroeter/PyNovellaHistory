system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')

import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix

import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ast import literal_eval
import numpy as np


def replace_full_name(name, list_of_names):
    try:
        matching = [s for s in list_of_names if name in s][0]
    except:
        matching = name
    return matching

scaler = StandardScaler()
scaler = MinMaxScaler()
columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"embedding_Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe", "Brustwarzen": "Erotik", "Geistwesen":"Spuk"}


infile_path = os.path.join(local_temp_directory(system), "DocDangerTypesMatrix_novellas_episodes.csv")
metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv" )

sent_fear_path = os.path.join(local_temp_directory(system), "DocSentFearMatrix_novellas_from-paragraph-chunks.csv")

spuk_path = os.path.join(global_corpus_representation_directory(system), "Spuk_DocThemesMatrix_novellas_episodes.csv")
matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)
df = matrix.data_matrix_df
df_doc_danger = df.copy()

sent_fear_df = pd.read_csv(sent_fear_path, index_col=0)
spuk_df = pd.read_csv(spuk_path, index_col=0)
df_concat1 = pd.concat([df, spuk_df], axis=1)
df = pd.concat([df_concat1, sent_fear_df], axis=1)

df = df.rename(columns=columns_transl_dict)
scaled_features = scaler.fit_transform(df)
df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


danger_df = df.drop(columns=["Erotik", "Liebe", "SentLexFear", "embedding_Angstempfinden","UnbekannteEindr", "Angstempfinden", "großzügig", "verwerflich", "Sturm", "Feuer", "Dämon"])

danger_df["max_value"] = danger_df.max(axis=1)
danger_df["max_danger_typ"] = danger_df.idxmax(axis=1)

danger_df["embedding_Angstempfinden"] = df["embedding_Angstempfinden"]
danger_df["Angstempfinden"] = df["Angstempfinden"]
danger_df["UnbekannteEindr"] = df["UnbekannteEindr"]
danger_df["Liebe"] = df["Liebe"]
danger_df["Erotik"] = df["Erotik"]
danger_df["Sturm"] = df["Sturm"]
danger_df["Feuer"] = df["Feuer"]

danger_df["doc_chunk_id"] = danger_df.index
#danger_df["doc_chunk_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[:-4])
danger_df["doc_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[:-5])
danger_df["chunk_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[-4:])



idx = danger_df.groupby("doc_id")["max_value"].transform(max) == danger_df["max_value"]
print(idx)
max_chunk_danger_df = danger_df[idx]

max_chunk_danger_df.set_index("doc_chunk_id", inplace=True)
#max_chunk_danger_df = max_chunk_danger_df.drop(columns=["doc_chunk_id"])



endanger_char_df = pd.read_csv(os.path.join(local_temp_directory(system), "DocNamesCounterMatrix_novellas_episodes.csv"), index_col=0)
max_danger_character_df = pd.concat([max_chunk_danger_df, endanger_char_df], axis=1)
max_danger_character_df.rename(columns={"0":"EndCharName"}, inplace=True)

max_danger_character_df.set_index("doc_id", inplace=True)

print(max_danger_character_df)

max_danger_character_df.drop_duplicates(inplace=True)

max_danger_character_df = max_danger_character_df[~max_danger_character_df.index.duplicated(keep='first')]

#characters_sna_df = pd.read_csv(os.path.join(global_corpus_representation_directory(system), "Heftromane_SNA_Emo_Matrix.csv"), index_col=0)
#max_danger_character_df = pd.concat([max_danger_character_df, characters_sna_df], axis=1)

df = max_danger_character_df
#df["EndCharName_full"] = df.apply(lambda x: replace_full_name(str(x.EndCharName), str(x.Figuren).split(". ")), axis=1)

df.dropna(subset=["max_value"], inplace=True)

from preprocessing.hardcoded_lists import male_list, female_list
male_list, female_list = male_list(), female_list()


#df["symp_EndChar"] = df.apply(lambda x: literal_eval(x.symp_dict)[x.EndCharName_full] if x.EndCharName_full in literal_eval(x.symp_dict) else np.nan, axis=1)
#df["centr_EndChar"] = df.apply(lambda x: literal_eval(x.deg_centr)[x.EndCharName_full] if x.EndCharName_full in literal_eval(x.deg_centr) else np.nan, axis=1)

#df["weigh_centr_EndChar"] = df.apply(lambda x: dict(literal_eval(x.weighted_deg_centr))[x.EndCharName_full] if x.EndCharName_full in dict(literal_eval(x.weighted_deg_centr)) else np.nan, axis=1)

#scaler = MinMaxScaler()
#scaled_weight_centr_endchar = scaler.fit_transform(df.weigh_centr_EndChar.values.reshape(-1,1))
#print(scaled_weight_centr_endchar)
#df["weigh_centr_EndChar"] = scaled_weight_centr_endchar

#df["gender_EndChar"] = df.apply(lambda x: "male" if x.EndCharName_full in male_list else ("female" if x.EndCharName_full in female_list else "unknown" ), axis=1)

print(df)
df.to_csv(os.path.join(local_temp_directory(system), "MaxDangerFearCharacters_novellas_episodes_scaled.csv"))

print(danger_df.max_danger_typ.values)

danger_df.to_csv(os.path.join(local_temp_directory(system), "AllChunksDangerFearCharacters_novellas_episodes_scaled.csv"))

print("finished!")