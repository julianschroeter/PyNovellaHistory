system = "my_xps" #"wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/PyNovellaHistory')

import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr, siegelslopes
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np

year_cat_name = "Erscheinungsjahr"
genre_cat_name = "Gattungen"
media_cat = "Medium"

novellas_infilepath = os.path.join(local_temp_directory(system), "MaxDangerFearCharactersHardseeds_novellas_Ganztexte_scaled.csv")
novellas_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

novellas_df = pd.read_csv(novellas_infilepath, index_col = 0)
novellas_dtm_obj = DocFeatureMatrix(data_matrix_filepath=novellas_infilepath, metadata_csv_filepath=novellas_metadata_filepath)
novellas_dtm_obj = novellas_dtm_obj.add_metadata(["Titel", "Nachname","Jahr_ED","Gattungslabel_ED_normalisiert","Medium_ED", "Kanon_Status", "seriell", "in_Deutscher_Novellenschatz"])
novellas_df = novellas_dtm_obj.data_matrix_df
novellas_df_full_media = standardize_meta_data_medium(df=novellas_df, medium_column_name="Medium_ED")

novellas_df = novellas_df_full_media.drop(columns=["Medium_ED", "medium"])
canon_cat = "canonicity"
novellas_df = novellas_df.rename(columns={"Titel": "title", "Nachname":"author", "Jahr_ED":"Erscheinungsjahr",
                                          "Gattungslabel_ED_normalisiert":"Gattungen", "medium_type": "Medium",
                                          "seriell":"Serialität", "Kanon_Status":canon_cat})

df = novellas_df

labels_list = ["R", "E", "N", "0E", "XE", "M"]
df = df[df.isin({genre_cat_name: labels_list}).any(axis=1)]

replace_dict = {genre_cat_name: {"N": "Novelle", "E": "Erzählung", "0E": "sonst. MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen", "XE": "sonst. MLP"}}

df = full_genre_labels(df, replace_dict=replace_dict)

replace_dict = {"Serialität": {"True": "Serie", "TRUE": "Serie", "vermutlich": "Serie",
                                                  "False": "nicht-seriell", "FALSE":"nicht-seriell"}}

df = full_genre_labels(df, replace_dict=replace_dict)


replace_dict = {canon_cat: {0: "low", 1: "low", 2: "high",
                                                  3:"high"}}
df = full_genre_labels(df, replace_dict=replace_dict)


whole_df = df.copy()
serial_status_list = ["Serie", "nicht-seriell"]


df = whole_df

genres_list = ["N", "E", "R", "0E", "XE", "M", "0P", "0PA", "0X_Essay", "0PB"]

genres_list = ["MLP", "Märchen"] #, "Spannungs-Heftroman", "Roman"
genres_list = ["Novelle", "Erzählung", "sonst. MLP", "Roman", "Märchen"]


media_list = ["Pantheon", "Journal", "Taschenbuch", "Familienblatt", "Rundschau", "Anthologie"]
media_list = ["Familienblatt", "Rundschau"]
media_list = ["Taschenbuch", "Pantheon"]

media_df = df[df.isin({media_cat: media_list}).any(axis=1)]
df = df[df.isin({genre_cat_name: genres_list}).any(axis=1)]
df_serial = df[df.isin({"Serialität": ["Serie"]}).any(axis=1)]
df_nonserial = df[df.isin({"Serialität": ["nicht-seriell"]}).any(axis=1)]
df_canon = df[df.isin({"Kanon_Status": ["hoch"]}).any(axis=1)]

colors_list = ["cyan", "yellow", "pink", "blue", "green", "yellow", "cyan", "cyan", "cyan", "cyan"] # for genres
media_colors_list = ["darkgreen", "lightblue","pink", "grey",  "cyan", "red", "green", "yellow", "orange", "blue", "magenta", "black"]
colors_list = ["red", "green", "cyan", "blue", "orange" ]
genres_dict = dict(zip(genres_list, colors_list[:len(genres_list)]))

media_dict = dict(zip(media_list, media_colors_list[:len(media_list)]))
media_mpatches_list = []
for genre, color in media_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    media_mpatches_list.append(patch)

media_colors_list = [media_dict[x] for x in media_df[media_cat].values.tolist()]


genres_mpatches_list = []

for genre, color in genres_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    genres_mpatches_list.append(patch)


genre_colors_list = [genres_dict[x] for x in df[genre_cat_name].values.tolist()]



canon_mpatches_list = []
for serial, color in {"hoch":"purple", "niedrig":"grey"}.items():
    patch = mpatches.Patch(color=color, label=serial)
    canon_mpatches_list.append(patch)

y_variables = ["Hardseeds", "Farben", "Abstrakta"]
# y_variable = "Farben" #"Abstrakta" # "Hardseeds" # "fear_love" #"Liebe" # "max_value" #"Angstempfinden" #  "lin_susp_model"
x_variable = year_cat_name  # , "UnbekannteEindr"

fig, axes = plt.subplots(3,2,figsize=(10,15))
i = 0
for y_variable in y_variables:
   if y_variable == "Farben":
        y_variable_legend = "Colors"
   elif y_variable == "Abstrakta":
        y_variable_legend = "Abstract Concepts"
   else:
        y_variable_legend = "Hardseeds"

   sns.lineplot(data=df, x=year_cat_name, y=y_variable, hue=canon_cat,
                 palette={"low":"grey", "high": "purple"}, ax=axes[i,0])
   axes[i,0].set_xlabel("")
   axes[i,0].set_ylabel(y_variable_legend)
   axes[i,0].set_ylim(0,1)

   regr = LinearRegression()
   regr.fit(df_serial.loc[:, x_variable].array.reshape(-1, 1), df_serial.loc[:, y_variable])
   y_pred = regr.predict(df_serial.loc[:, x_variable].array.reshape(-1, 1))
   #plt.plot(df_serial.loc[:, x_variable], y_pred, color="black", linewidth=1, linestyle=":")
   # siegel-slope
   x = df.loc[:, x_variable]
   res = siegelslopes(df.loc[:, y_variable], x)
   axes[i,1].plot(x, res[1] + res[0] * x, color="grey", linewidth=3)

   axes[i,1].scatter(df.loc[:, x_variable], df.loc[:, y_variable], color="grey", alpha=0.5)

   x = df.loc[:, x_variable]
   res = siegelslopes(df.loc[:, y_variable], x)
   axes[i,1].plot(x, res[1] + res[0] * x, color="black", linewidth=3)
   i += 1

axes[0,0].set_title("Lineplot")
axes[0,1].set_title("Scatterplot and Siegel-Regression")
fig.supxlabel("Year of first publication")
fig.suptitle("Timeline for Hardseeds versus Abstract Concepts")
fig.tight_layout()

outfilename = "plot_hardseeds_timeline.svg"
fig.savefig(os.path.join(local_temp_directory(system), "figures", outfilename))
fig.show()

print("Finished!")