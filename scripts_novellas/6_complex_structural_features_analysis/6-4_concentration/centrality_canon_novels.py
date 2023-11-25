system = "my_xps"#  "wcph113" # "my_mac" # "wcph104"

from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, global_corpus_representation_directory, load_stoplist, local_temp_directory
from preprocessing.sampling import equal_sample
from preprocessing.sna import get_centralization
from preprocessing.corpus import DocFeatureMatrix
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = "Networkdata_document_Matrix.csv" # "SNA_novellas.csv"
data_matrix_filepath = os.path.join(global_corpus_representation_directory(system), filename)
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system), "Bibliographie.csv")
textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "textanalytic_metadata.csv")

matrix = DocFeatureMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                 metadata_csv_filepath = textanalytic_metadata_filepath, mallet=False)

matrix = matrix.add_metadata(["cluster_komplexität", "figurenanzahl", "strängigkeit"])
df = matrix.data_matrix_df
df = df.rename(columns={"Figurenanzahl": "Figurenanzahl_rule", "Netzwerkdichte":"Netzwerkdichte_rule", "centralization":"centralization_rule", "weighted_deg_centr" : "weighted_deg_rule",
                        "Anteil Figuren mit degree centrality == 1":"Anteil_deg_centr_1_rule", "deg_centr":"deg_centr_rule", "Figuren": "Figuren_rule"
                        })


filename = "conll_based_networkdata-matrix-novellas_15mostcommon.csv"

data_matrix_filepath = os.path.join(local_temp_directory(system), filename)
matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=df,
                          metadata_df=None,
                                  metadata_csv_filepath = data_matrix_filepath, mallet=False)

matrix = matrix.add_metadata(["degree_centrality", "density", "centralization", "Figuren", "Figurenanzahl", "weighted_deg_centr" ])

df = matrix.data_matrix_df
df = df.rename(columns={"Figurenanzahl": "Figurenanzahl_conll", "figurenanzahl":"Figurenanzahl_annotationen", "centralization": "centralization_conll",
                        "degree_centrality":"deg_centr_conll", "density": "Netzwerkdichte_conll", "weighted_deg_centr" :"weighted_deg_conll",
                        "Figuren":"Figuren_conll", "Figurenanzahl":"Figurenanzahl_conll"})

metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system), "Bibliographie.csv")
matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=df, metadata_df=None,
                                 metadata_csv_filepath = metadata_csv_filepath, mallet=False)
matrix = matrix.add_metadata(["Nachname", "Titel", "Gattungslabel_ED_normalisiert", "Jahr_ED", "Medientyp_ED", "in_Deutscher_Novellenschatz", "Kanon_Status"])

matrix = matrix.reduce_to_categories("Gattungslabel_ED_normalisiert", ["N", "E", "0E", "XE", "R"])

df = matrix.data_matrix_df
df.loc[:,"Netzwerkdichte_conll"] = df.loc[:,"Netzwerkdichte_conll"].fillna(0)
df.loc[:,"centralization_conll"] = df.loc[:,"centralization_conll"].fillna(0)
df.loc[:,"Figurenanzahl_conll"] = df.loc[:,"Figurenanzahl_conll"].fillna(0)
df.loc[:,"deg_centr_conll"] = df.loc[:,"deg_centr_conll"].apply(lambda x: "{}" if x == "{'NO_CHARACTER': 0}" else x)

df.loc[:,"deg_centr_conll"] = df.loc[:,"deg_centr_conll"].fillna("{}")
df.loc[:,"weighted_deg_conll"] = df.loc[:,"weighted_deg_conll"].fillna("{}")

df.loc[:,"strängigkeit"] =df.loc[:,"strängigkeit"].apply(lambda x: str(x)[:7])

df["density_combined"] = df.apply(lambda x: (x["Netzwerkdichte_rule"] + x["Netzwerkdichte_conll"]) / 2, axis=1)

df = df[df["density_combined"] != 0]
df = df[df["Figurenanzahl_conll"] > 1]

df["centralization_combined"] = df.apply(lambda x: (x["centralization_rule"] + x["centralization_conll"]) / 2, axis=1)

df["dens_centr"] = df.apply(lambda x: (x["centralization_combined"] * x["density_combined"]), axis=1)

df["Figurenanzahl_combined"] = df.apply(lambda x: (x["Figurenanzahl_rule"] + x["Figurenanzahl_conll"]) / 2, axis=1)

def scale_centrality(degree_centr_dict, weighted_dict):
    scaled_centrality_dict = {}
    for key, value in degree_centr_dict.items():
        scaled_centrality_dict[key] = weighted_dict[key] / max(weighted_dict.values()) * degree_centr_dict[max(weighted_dict, key=weighted_dict.get)]
    return scaled_centrality_dict

df["scaled_centrality"] = df.apply(lambda x: scale_centrality(eval(x["deg_centr_rule"]), dict(eval(x["weighted_deg_rule"]))), axis=1)
df["scaled_centralization_rule"] = df.apply(lambda x: get_centralization(dict(x["scaled_centrality"]), c_type="degree"), axis=1)

df["scaled_centrality_conll"] = df.apply(lambda x: scale_centrality(eval(x["deg_centr_conll"]), dict(eval(x["weighted_deg_conll"]))), axis=1)
df["scaled_centralization_conll"] = df.apply(lambda x: get_centralization(dict(x["scaled_centrality_conll"]), c_type="degree"), axis=1)

df["scaled_centralization_combined"] = df.apply(lambda x: (x["centralization_rule"] + x["centralization_conll"]) / 2, axis=1)


df.boxplot(column="centralization_conll", by="Gattungslabel_ED_normalisiert")
plt.show()


df.boxplot(column="Netzwerkdichte_conll", by="Gattungslabel_ED_normalisiert")
plt.show()

