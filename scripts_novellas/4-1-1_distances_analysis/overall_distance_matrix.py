system = "my_xps" #"wcph113"

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import distance_matrix
import os
import pandas as pd

from preprocessing.corpus import DTM, DocFeatureMatrix
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory
from classification.perspectivalmodeling import split_features_labels

label_list = ["R", "M", "E", "N", "0E", "XE"]
metric="cosine"

filename = "red-to-2500mfw_red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)

dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert"])
dtm_obj = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=label_list)
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge"])

df = dtm_obj.data_matrix_df
df_dtm = df.copy()
df = df.drop(columns=["Gattungslabel_ED_normalisiert"])
dist_matrix = pdist(df, metric=metric)
v = squareform(dist_matrix)
dist_matrix = pd.DataFrame(v, index=df.index, columns=df.index)
print(dist_matrix)
dist_matrix.to_csv(os.path.join(global_corpus_representation_directory(system), "overall_distances_matrix.csv"))


# explore distances for individual texts:
metadata = pd.read_csv(metadata_path, index_col=0)
df = dist_matrix

alexis = df.loc[:,["00382-00"]]
print(alexis)
alexis = alexis.sort_values(by=["00382-00"])

obj = DocFeatureMatrix(data_matrix_filepath=None, metadata_csv_filepath=metadata_path, data_matrix_df=alexis)
obj = obj.add_metadata(["Nachname", "Titel", "Gattungslabel_ED_normalisiert", "Medientyp_ED", "Jahr_ED"])
results = obj.data_matrix_df

results = results[results["Jahr_ED"] >= 1870]
print(results)