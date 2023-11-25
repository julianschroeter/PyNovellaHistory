system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# from my own modules:
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods

# standard libraries
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import cov
from scipy.stats import pearsonr, spearmanr, siegelslopes
import seaborn as sns



medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"


old_infile_name = os.path.join(global_corpus_representation_directory(system), "SNA_novellas.csv")
filepath = os.path.join(global_corpus_representation_directory(system), "Network_Matrix_all.csv")

metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


matrix_obj = DocFeatureMatrix(data_matrix_filepath=filepath, metadata_csv_filepath= metadata_filepath)
matrix_obj = matrix_obj.add_metadata([genre_cat, year_cat, medium_cat, "Nachname", "Titel", "Kanon_Status"])

df1 = matrix_obj.data_matrix_df


length_infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix_obj.data_matrix_df,
                              metadata_csv_filepath=length_infile_df_path)
matrix_obj = matrix_obj.add_metadata(["token_count"])

cat_labels = ["N", "E"]
cat_labels = ["0E", "XE"]
cat_labels = ["N"]
cat_labels = ["E"]
cat_labels = ["N", "E", "0E", "XE"]
cat_labels = ["R"]
cat_labels = ["N", "E", "0E", "XE", "M", "R"]

matrix_obj = matrix_obj.reduce_to_categories(genre_cat, cat_labels)

matrix_obj = matrix_obj.eliminate(["Figuren"])

df = matrix_obj.data_matrix_df
df = df[~df["token_count"].isna()]
#df = df[~df["Netzwerkdichte_rule"].isna()]
df = df[df["Figurenanzahl_conll"]>0]
df.rename(columns={"scaled_centralization_conll": "Zentralisierung"}, inplace=True)
df.rename(columns={"Netzwerkdichte_conll": "Netzwerkdichte"}, inplace=True)

df = df[~df["Zentralisierung"].isna()]


print(df["Zentralisierung"].dtypes)
df["Zentralisierung"] = df["Zentralisierung"].astype(float)

print("Covariance. ", cov(1-df["Zentralisierung"], df["token_count"]))

corr, _ = pearsonr(1- df["Zentralisierung"], df["token_count"])
print('Pearsons correlation: %.3f' % corr)

corr, _ = spearmanr(1- df["Zentralisierung"], df["token_count"])
print('Spearman correlation: %.3f' % corr)

df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1800, end_year=1950, epoch_length=50,
                      new_periods_column_name="periods")


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "MLP",
                                    "R": "Roman", "M": "Märchen", "XE": "MLP"}}
df = full_genre_labels(df, replace_dict=replace_dict)

#df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
 #                            , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)


colors_list = ["red", "green", "cyan","orange", "blue"]
genres_list = df[genre_cat].values.tolist()
list_targetlabels = ", ".join(map(str, set(genres_list))).split(", ")

zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))


zipped_dict = {"Novelle":"red", "Erzählung":"green", "MLP": "cyan", "Märchen":"orange", "Roman":"blue"}
list_colors_target = [zipped_dict[item] for item in genres_list]

bildungsroman_ids = ["00549-00", "00557-00", "00558-00",
                                             "00551-00"]


bild_romane = df.loc[bildungsroman_ids]
romane = df[df["Gattungslabel_ED_normalisiert"]=="Roman"]
novellen = df[df["Gattungslabel_ED_normalisiert"] == "Novelle"]
plt.boxplot([romane["Zentralisierung"], bild_romane["Zentralisierung"], novellen["Zentralisierung"]])
plt.xticks([1,2,3], ["Romane", "Bildungsromane", "Novellen"])
plt.title("Gattungsvergleich: Zentralisierung")
plt.ylabel("Zentralisierung")
plt.show()
