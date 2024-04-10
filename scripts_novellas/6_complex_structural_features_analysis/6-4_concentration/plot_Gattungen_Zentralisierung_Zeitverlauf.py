system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# from my own modules:
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods

import os
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import siegelslopes

medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"


old_infile_name = os.path.join(global_corpus_representation_directory(system), "SNA_novellas.csv")
infile_name = os.path.join(global_corpus_representation_directory(system), "Network_Matrix_all.csv")
print(infile_name)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)
#matrix_obj = matrix_obj.add_metadata([genre_cat, year_cat, medium_cat, "Nachname", "Titel"])

df1 = matrix_obj.data_matrix_df

matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix_obj.data_matrix_df, metadata_csv_filepath=old_infile_name)
matrix_obj = matrix_obj.add_metadata(["Netzwerkdichte"])

length_infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix_obj.data_matrix_df,
                              metadata_csv_filepath=length_infile_df_path)
matrix_obj = matrix_obj.add_metadata(["token_count"])


cat_labels = ["N", "E"]
cat_labels = ["N", "E", "0E", "XE", "R", "M"]
matrix_obj = matrix_obj.reduce_to_categories(genre_cat, cat_labels)

matrix_obj = matrix_obj.eliminate(["Figuren"])

df = matrix_obj.data_matrix_df

df.rename(columns={"scaled_centralization_conll": "dep_var"}, inplace=True) # "Netzwerkdichte"

df = df[~df["token_count"].isna()]


dep_var = "dep_var"
df.rename(columns={"Zentralisierung": "dep_var","Gattungslabel_ED_normalisiert" : "Gattung_ED",
                   "Netzwerkdichte_conll":"Netzwerkdichte"}, inplace=True) # "Netzwerkdichte"
df["dep_var"] = df["dep_var"].astype(float)



df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1770, end_year=1970, epoch_length=20,
                      new_periods_column_name="periods")


replace_dict = {"Gattung_ED": {"N": "Novelle", "E": "Erzählung", "0E": "Prosa",
                                    "R": "Prosa", "M": "Prosa", "XE": "Prosa"}}
df = full_genre_labels(df, replace_dict=replace_dict)


#df = df[df.isin({medium_cat:["Familienblatt", "Anthologie", "Taschenbuch", "Rundschau"]}).any(1)]
df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)

df = df[df["Jahr_ED"] >= 1810]

x = df.loc[:, "Jahr_ED"]
res = siegelslopes(df.loc[:, "dep_var"], x)
sns.lineplot(data=df, x="Jahr_ED", y="dep_var", hue="Gattung_ED",
             palette=["red", "grey","green" ])
plt.plot(x, res[1] + res[0] * x, color="grey", linewidth=1)

N_df = df[df["Gattung_ED"] == "Novelle"]
x = N_df.loc[:, "Jahr_ED"]
res = siegelslopes(N_df.loc[:, "dep_var"], x)
plt.plot(x, res[1] + res[0] * x, color="red", linewidth=1)
plt.ylim(0,1)
plt.title("Zentralisierung in Novellen und übriger Prosa")
plt.ylabel("Zentralisierung")
plt.savefig(os.path.join(local_temp_directory(system), "figures", "Abb_Zentralisierung_in_Novellen_vs_sonstige_Prosa.svg"))
plt.show()