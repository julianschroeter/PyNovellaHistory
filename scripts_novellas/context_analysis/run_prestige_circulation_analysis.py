import os

import matplotlib.pyplot as plt

from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory, load_stoplist, vocab_lists_dicts_directory
from preprocessing.metadata_transformation import full_genre_labels, years_to_periods, generate_media_dependend_genres
from preprocessing.corpus_alt import DocFeatureMatrix
import pandas as pd

system = "wcph113"



infile_name = os.path.join(global_corpus_raw_dtm_directory(system), "no-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv") # hier die Eingangsdatei der Featurerepräsentation wählen
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "my_colors.txt"))

rel_metadata = ["Gattungslabel_ED_normalisiert", "Gender", "Jahr_ED", "Medientyp_ED", "Kanon_Status"]

df_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)
df_obj = df_obj.add_metadata(rel_metadata)
new_df_obj = df_obj.reduce_to(rel_metadata, return_eliminated_terms_list=False)
new_df_obj = new_df_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=["N", "E", "0E", "XE"])

start_df = new_df_obj.data_matrix_df

periods_df = years_to_periods(input_df=start_df, category_name="Jahr_ED",
                              start_year=1790, end_year=2000, epoch_length=20,
                              new_periods_column_name="periods30a")

replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "sonstige Prosaerzählung", "0R" : "sonstige Journalinhalte", "0PE" : "sonstige Prosaerzählung", "0X_Essay" : "sonstige Journalinhalte", "0PB" : "sonstige Journalinhalte", "Lyrik" : "sonstige Journalinhalte", "Drama" : "sonstige Journalinhalte",
                                    "R": "Roman", "M": "Märchen", "XE": "sonstige Prosaerzählung", "V": "sonstige Journalinhalte", "V ": "sonstige Journalinhalte", "0P": "sonstige Journalinhalte", "0": "sonstige Journalinhalte"}}


full_genres_df = full_genre_labels(periods_df, replace_dict=replace_dict)

processing_df = full_genres_df.copy()




replace_dict = {"Medientyp_ED": {"Zeitschrift": 1000, "Zeitung": 1000, "Kalender": 5000, "Rundschau" : 5000,
                                 "Zyklus" : 250, "Roman" : 250, "(unbekannt)" : 250,
                                    "Illustrierte": 10000, "Sammlung": 250, "Nachlass": 10,
                                 "Familienblatt" : 100000, "Anthologie":500, "Taschenbuch": 250,
                                 "Werke":500,
                                 "Jahrbuch":250, "Monographie": 250}}


volumes_df = full_genre_labels(processing_df, replace_dict=replace_dict)

df = volumes_df.drop(columns=["Jahr_ED", "Kanon_Status", "Gender"])
df = df.rename( {"Medientyp_ED" : "counts"}, axis=1)

grouped = df.groupby(["periods30a", "Gattungslabel_ED_normalisiert"]).mean()
df_grouped = pd.DataFrame(grouped)
print(df_grouped)
df = df_grouped
print(df)

x_ticks = ["1790-1810", "1810-1830", "1830-1850", "1850-1870", "1870-1890",
 "1890-1910", "1910-1930", "1930-1850"]

fig, ax = plt.subplots()
ax = df["counts"].unstack().plot(kind='line', stacked=False, title=str("Temporal development of circulation for genres"),
                                       ylabel=str("circulation: volume (log scale)"), xlabel="time",
                                           )

ax.set_xticks(range(len(x_ticks)))
ax.set_xticklabels(["1790", "1810","1830","1850","1870","1890", "1910", "1930"])
plt.yscale("log")
plt.show()


df = volumes_df.drop(columns=["Jahr_ED", "Medientyp_ED", "Gender"])
df = df.rename( {"Kanon_Status" : "counts"}, axis=1)

grouped = df.groupby(["periods30a", "Gattungslabel_ED_normalisiert"]).mean()
df_grouped = pd.DataFrame(grouped)
print(df_grouped)
df = df_grouped
print(df)
fig, ax = plt.subplots()
ax = df["counts"].unstack().plot(kind='line', stacked=False, title=str("Temporal development of canonicity for genres"),
                                       ylabel=str("canonicity (0-3"), xlabel="time",
                                           )
ax.set_xticks(range(len(x_ticks)))
ax.set_xticklabels(["1790", "1810","1830","1850","1870","1890", "1910", "1930"])
plt.show()

df = start_df.copy()
df = years_to_periods(input_df=df, category_name="Jahr_ED",
                              start_year=1790, end_year=2000, epoch_length=10,
                              new_periods_column_name="periods")
df = df.drop(columns=["Gender", "Jahr_ED"])
# df = df.rename( {"Medientyp_ED" : "counts"}, axis=1)

df = generate_media_dependend_genres(df)

print(df)

#periods = ["1810-1830","1830-1850", "1850-1870", "1870-1890"]
#df = df[df.isin({"periods30a": periods}).any(1)]

grouped = df.groupby(["periods", "dependent_genre"]).mean()
df_grouped = pd.DataFrame(grouped)
df = df_grouped
print(df)
fig, ax = plt.subplots()
ax = df["Kanon_Status"].unstack().plot(kind='line', stacked=False, title=str("Temporal development of canonicity for media dependent genres"),
                                       ylabel=str("canonicity (0-3)"), xlabel="time",
                                           )
#ax.set_xticks(range(len(x_ticks)))
#ax.set_xticklabels(["1790", "1810","1830","1850","1870","1890", "1910", "1930"])
plt.show()
