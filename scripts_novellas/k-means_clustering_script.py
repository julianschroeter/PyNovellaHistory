system = "wcph113"

from sklearn.cluster import KMeans
import os

from preprocessing.corpus_alt import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory
from classification.perspectivalmodeling import split_features_labels

label_list = ["R", "M"]

filename = "RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)

dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert"])
dtm_obj = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=label_list)
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge"])

df = dtm_obj.data_matrix_df

print(df)

X, Y = split_features_labels(df)

k = 2

kmeans = KMeans(n_clusters=k)

y_pred = kmeans.fit_predict(X)

