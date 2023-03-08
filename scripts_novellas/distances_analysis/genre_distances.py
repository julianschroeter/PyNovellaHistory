system = "wcph113"

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import distance_matrix
import os
import pandas as pd

from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory
from classification.perspectivalmodeling import split_features_labels

label_list = ["R", "M", "E", "N", "0E", "XE"]

filename = "scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)

dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert"])
dtm_obj = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=label_list)
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge"])

df = dtm_obj.data_matrix_df

df_N = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=["N"]).eliminate(["Gattungslabel_ED_normalisiert"]).data_matrix_df
df_E = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=["E"]).eliminate(["Gattungslabel_ED_normalisiert"]).data_matrix_df
df_M = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=["M"]).eliminate(["Gattungslabel_ED_normalisiert"]).data_matrix_df
df_R = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=["R"]).eliminate(["Gattungslabel_ED_normalisiert"]).data_matrix_df


print(df_N)



print(df_E)


print("for Novellen: ")
distances = pdist(df_N, metric="cosine")
print("Average distance: ", distances.mean())
print("Std: ", distances.std())

v = squareform(distances)
print(v)
dist_matr_df = pd.DataFrame(v, index = df_N.index, columns= df_N.index)
dist_matr_df["mean_dist"] = dist_matr_df.sum(axis=0)
print(dist_matr_df.sort_values(by="mean_dist", axis=0))

print("for Erzählungen: ")
distances = pdist(df_E, metric="cosine")
print("Average distance: ", distances.mean())
print("Std: ", distances.std())

print("for Märchen: ")
distances = pdist(df_M, metric= "cosine")
print("Average distance: ", distances.mean())
print("Std: ", distances.std())


print("for Novels: ")
distances = pdist(df_R, metric= "cosine")
print("Average distance: ", distances.mean())
print("Std: ", distances.std())


print("metrics between Novellen and Erzählungen:")
distances = cdist(df_N, df_E, metric="cosine")
print("Average distance: ", distances.mean())
print("Std: ", distances.std())


dist_matrix = distance_matrix(df_N, df_N, p=2)
print(dist_matrix)

