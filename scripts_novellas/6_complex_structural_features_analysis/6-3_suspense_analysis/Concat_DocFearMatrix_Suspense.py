system = "my_xps"#"wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix

import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler



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


infile_path = os.path.join(global_corpus_representation_directory(system), "DocDangerTypesMatrix_episodes.csv")
metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv" )

sent_fear_path = os.path.join(global_corpus_representation_directory(system), "SentFearMatrix_episodes.csv")

matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)
df = matrix.data_matrix_df

print(df)

sent_fear_df = pd.read_csv(sent_fear_path, index_col=0)
df = pd.concat([df, sent_fear_df], axis=1)

df = df.rename(columns=columns_transl_dict)
scaled_features = scaler.fit_transform(df)
df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

print(df)

danger_df = df.drop(columns=["Liebe", "SentLexFear", "embedding_Angstempfinden","UnbekannteEindr", "Angstempfinden", "großzügig", "verwerflich", "plötzlich"])

danger_df["max_value"] = danger_df.max(axis=1)
danger_df["max_danger_typ"] = danger_df.idxmax(axis=1)
danger_df["embedding_Angstempfinden"] = scaler.fit_transform(df["embedding_Angstempfinden"].values.reshape(-1,1))
danger_df["Angstempfinden"] = scaler.fit_transform(df["Angstempfinden"].values.reshape(-1,1))
danger_df["UnbekannteEindr"] = scaler.fit_transform(df["UnbekannteEindr"].values.reshape(-1,1))
danger_df["Liebe"] = scaler.fit_transform(df["Liebe"].values.reshape(-1,1))
danger_df["plötzlich"] = scaler.fit_transform(df["plötzlich"].values.reshape(-1,1))
danger_df["doc_chunk_id"] = danger_df.index
#danger_df["doc_chunk_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[:-4])
danger_df["doc_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[:-5])
danger_df["chunk_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[-4:])

print(danger_df)




endanger_char_df = pd.read_csv(os.path.join(local_temp_directory(system), "DocNamesCounterMatrix_novellas.csv"), index_col=0)

print(endanger_char_df)

max_danger_character_df = pd.concat([danger_df, endanger_char_df], axis=1)
max_danger_character_df.rename(columns={"0":"EndCharName"}, inplace=True)

max_danger_character_df.set_index("doc_id", inplace=True)

print(max_danger_character_df)

max_danger_character_df.drop_duplicates(inplace=True)


max_danger_character_df = max_danger_character_df[~max_danger_character_df.index.duplicated(keep='first')]



danger_df.to_csv(os.path.join(local_temp_directory(system), "All_Chunks_Danger_FearCharacters_episodes.csv"))