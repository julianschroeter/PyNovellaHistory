import pandas as pd
import os
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory, load_stoplist
from preprocessing.metadata_transformation import full_genre_labels
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from clustering.my_plots import scatter
from preprocessing.corpus import DocFeatureMatrix
from collections import Counter

system = "wcph113"
infile_name = os.path.join(global_corpus_raw_dtm_directory(system), "DocThemesMatrix_genres.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
colors_list = load_stoplist(os.path.join(global_corpus_representation_directory(system), "my_colors.txt"))

df_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)
df = df_obj.data_matrix_df

scaler = StandardScaler()

df.boxplot(by="Gattungslabel_ED_normalisiert", figsize=[20,20])
plt.show()

new_df_obj = df_obj.reduce_to(["lieben", "Marseille", "Gattungslabel_ED_normalisiert"], return_eliminated_terms_list=False)
new_df = new_df_obj.data_matrix_df
replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "sonstige Prosaerzählung",
                                    "R": "Roman", "M": "Märchen", "XE": "sonstige Prosaerzählung"}}
new_df = full_genre_labels(new_df, replace_dict=replace_dict)
# scale:
new_df.iloc[:, :2] = scaler.fit_transform(new_df.iloc[:, :2].to_numpy())

print(new_df)
scatter(new_df, colors_list)



new_df_obj = df_obj.add_metadata(["Nachname", "Titel", "Medium_ED", "Jahr_ED"])
#new_df_obj = new_df_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=["E", "0E", "N", "XE"])
df = new_df_obj.data_matrix_df

after1850_df = df[df["Jahr_ED"] > 1850]

marseille_sorted_df = new_df_obj.data_matrix_df.sort_values(by="Marseille", ascending=False)

print("Verteilung der Gattungslabels: ", marseille_sorted_df["Gattungslabel_ED_normalisiert"].value_counts())

reduced_df = marseille_sorted_df[marseille_sorted_df["Marseille"] > 0.00004]

labels = reduced_df["Gattungslabel_ED_normalisiert"].values.tolist()

labels_counter = Counter(labels)
print(labels_counter)

reduced_df = reduced_df[reduced_df["Jahr_ED"] > 1850]

labels = reduced_df["Gattungslabel_ED_normalisiert"].values.tolist()

labels_counter = Counter(labels)
print(labels_counter)