system = "my_xps"

from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory, global_corpus_raw_dtm_directory, global_corpus_directory
from preprocessing.corpus import DocFeatureMatrix, generate_text_files
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

corpus_path = global_corpus_directory(system)
metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

metadata_df = pd.read_csv(metadata_filepath, index_col=0)
print(metadata_df)
df = metadata_df.copy()
df = df.drop(labels="00311-00")
BD_df = df[df["spätere_Fassung_von"].notna()]

# drop works by author "Poe"
BD_df = BD_df[BD_df["Nachname"] != "Poe"]
BD_ids = BD_df.index.values.tolist()
ED_ids = BD_df["spätere_Fassung_von"].values.tolist()
# FV: Fassungsvergleiche
FV_ids = ED_ids + BD_ids

ED_df = df.loc[ED_ids]
print((ED_df))
filename = "scaled_raw_dtm_l1__use_idf_False2500mfw.csv" # "raw_dtm_l2_lemmatized_use_idf_True2500mfw.csv" # "raw_dtm_l1__use_idf_False2500mfw.csv"
input_dtm_filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)
dtm_obj = DocFeatureMatrix(data_matrix_filepath=input_dtm_filepath, metadata_csv_filepath=metadata_filepath)
dtm_obj = dtm_obj.add_metadata(["Jahr_ED", "Nachname", "Gattungslabel_ED_normalisiert"])
data = dtm_obj.data_matrix_df

data_ED = data.loc[ED_ids]
data_BD = data.loc[BD_ids]

from metrics.distances import results_2groups_dist, InterGroupDistances

distances = InterGroupDistances(data_ED, data_BD, metric="cosine", select_one_per_author=False, select_one_per_period=False,
                 smaller_sample_size=False)


dist_matrix = distances.dist_matr_df
dist_matrix = dist_matrix.drop(labels="mean_dist")
dist_matrix = dist_matrix.drop(columns="mean_dist")

dist_matrix = dist_matrix.reindex(labels=ED_ids)
dist_matrix = dist_matrix.reindex(columns=BD_ids)

columns = dist_matrix.columns.tolist()
indexes = dist_matrix.index.values.tolist()
#dist_matrix["Werk"] = dist_matrix.apply(lambda x: df.loc[x.name, "Titel"], axis=1)

work_names = [df.loc[x, "Titel"] for x in columns]
col_work_dict = dict(zip(columns, work_names))
dist_matrix.rename(columns=col_work_dict, inplace=True)

work_names_ED = [df.loc[x, "Titel"] for x in indexes]
col_work_dict = dict(zip(indexes, work_names_ED))
dist_matrix.rename(index=col_work_dict, inplace=True)

print(dist_matrix)

sns.set(font_scale=0.5)
cmap = sns.cm.rocket_r
sns.heatmap(dist_matrix, annot=True, xticklabels=True, yticklabels=True,
            vmin=0.1, vmax=1.1, cmap="gray")
plt.title("Heatmap für Fassungsvergleiche")
plt.ylabel("Erstdruck")
plt.xlabel("Spät-/Buchdruck")
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures", "heatmap_Fassungsvergleiche_grey.svg"))
plt.show()

results = results_2groups_dist(1, data_ED, data_BD, metric="cosine", select_one_author=False,select_one_per_period=False)
print(results)