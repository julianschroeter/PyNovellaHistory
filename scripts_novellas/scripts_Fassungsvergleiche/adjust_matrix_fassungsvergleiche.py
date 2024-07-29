system =  "my_xps" # "wcph113"
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


scaler = MinMaxScaler()
scaler = StandardScaler()
columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"embedding_Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe", "Brustwarzen": "Erotik", "Geistwesen":"Spuk", "Wort,Klasse":"Hardseeds"}


infile_path = os.path.join(global_corpus_representation_directory(system), "DocDangerTypes_Fassungsvergleiche.csv")
metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv" )



matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)
df = matrix.data_matrix_df
df_doc_danger = df.copy()



df = df.rename(columns=columns_transl_dict)
scaled_features = scaler.fit_transform(df)
df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


danger_df = df.copy()

danger_df["max_value"] = danger_df.max(axis=1)
danger_df["max_danger_typ"] = danger_df.idxmax(axis=1)

danger_df["doc_chunk_id"] = danger_df.index
#danger_df["doc_chunk_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[:-4])
danger_df["doc_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[:-5])

print(danger_df)
danger_df.to_csv(os.path.join(global_corpus_representation_directory(system), "MaxDangerMatrix_Fassungsvergleiche.csv"))

print("finished!")