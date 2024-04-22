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


novellas_infilepath = os.path.join(local_temp_directory(system),  "AllChunksDangerFearCharacters_novellas_episodes_scaled.csv")


novellas_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")



novellas_df = pd.read_csv(novellas_infilepath, index_col = 0)
novellas_dtm_obj = DocFeatureMatrix(data_matrix_filepath=novellas_infilepath, metadata_csv_filepath=novellas_metadata_filepath)
novellas_dtm_obj = novellas_dtm_obj.add_metadata(["Titel", "Nachname","Jahr_ED","Gattungslabel_ED_normalisiert","Medium_ED", "Kanon_Status", "seriell"])
novellas_df = novellas_dtm_obj.data_matrix_df
novellas_df_full_media = standardize_meta_data_medium(df=novellas_df, medium_column_name="Medium_ED")

novellas_df = novellas_df_full_media.drop(columns=["Medium_ED", "medium"])

novellas_df = novellas_df.rename(columns={"Titel": "title", "Nachname":"author", "Jahr_ED":"Erscheinungsjahr",
                                          "Gattungslabel_ED_normalisiert":"Gattungen", "medium_type": "Medium",
                                          "seriell":"Serialität"})

df = novellas_df

df = df[df["doc_chunk_id"].map(len) == 8]
print(df)

labels_list = ["R", "M", "E", "N", "0E", "XE", "0P", "0PB", "krimi", "abenteuer", "krieg"]
labels_list = ["R", "E", "N", "0E", "XE", "M"]
df = df[df.isin({genre_cat_name: labels_list}).any(axis=1)]

other_cat_labels_list =  ["Taschenbuch", "Familienblatt", "Rundschau"]
other_cat_labels_list = ["Kleist", "Goethe", "Hoffmann" , "Eichendorff","Tieck", "Stifter", "Storm", "Keller", "Meyer", "Schnitzler", "Mann", "Musil"]
#whole_df = whole_df[whole_df.isin({"author": other_cat_labels_list}).any(1)]


replace_dict = {genre_cat_name: {"N": "MLP", "E": "MLP", "0E": "MLP", "XE": "MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen",
                          "krimi": "Spannungs-Heftroman", "abenteuer": "Spannungs-Heftroman", "krieg": "Spannungs-Heftroman"}}

replace_dict = {genre_cat_name: {"N": "Novelle", "E": "Erzählung", "0E": "sonst. MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen", "XE": "sonst. MLP"}}


df = full_genre_labels(df, replace_dict=replace_dict)

replace_dict = {"Serialität": {"True": "Serie", "TRUE": "Serie", "vermutlich": "Serie",
                                                  "False": "nicht-seriell", "FALSE":"nicht-seriell"}}

df = full_genre_labels(df, replace_dict=replace_dict)

#whole_df = df.drop(df['max_value'].idxmax())
whole_df = df.copy()
serial_status_list = ["Serie", "nicht-seriell"]
#whole_df = whole_df[whole_df.isin({"Serialität": serial_status_list}).any(axis=1)]

# coefficients for linear suspense model based on correlation in annotations: suspense = max_danger_level + 0.725 * Fear_level
whole_df["lin_susp_model"] = whole_df.apply(lambda x: x.max_value + (0.725 * x.Angstempfinden), axis=1)

scaler = MinMaxScaler()

scaled_values = scaler.fit_transform(whole_df[["Gewaltverbrechen", "Kampf", "Entführung", "Krieg", "Spuk","max_value", "Angstempfinden", "Sturm", "Feuer", "lin_susp_model"]])
whole_df[["Gewaltverbrechen", "Kampf", "Entführung", "Krieg","Spuk","max_value", "Angstempfinden", "Sturm", "Feuer", "lin_susp_model"]] = scaled_values
#whole_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

whole_df = years_to_periods(input_df=whole_df, category_name=year_cat_name, start_year=1770,end_year=1940, epoch_length=20,
                            new_periods_column_name="periods")

#whole_df = whole_df[whole_df["lin_susp_model"] != 0]
df = whole_df.copy()

danger_df = df.drop(columns=["Erotik", "Liebe", "embedding_Angstempfinden","UnbekannteEindr", "Angstempfinden", "Liebe", "Erotik"])

danger_df["max_value"] = danger_df[["Gewaltverbrechen", "Kampf", "Entführung","Krieg","Sturm","Feuer"]].max(axis=1)
danger_df["max_danger_typ"] = danger_df[["Gewaltverbrechen", "Kampf", "Entführung","Krieg","Sturm","Feuer"]].idxmax(axis=1)

danger_df["embedding_Angstempfinden"] = df["embedding_Angstempfinden"]
danger_df["Angstempfinden"] = df["Angstempfinden"]
danger_df["UnbekannteEindr"] = df["UnbekannteEindr"]
danger_df["Liebe"] = df["Liebe"]
danger_df["Erotik"] = df["Erotik"]
danger_df["Sturm"] = df["Sturm"]
danger_df["Feuer"] = df["Feuer"]



df = whole_df

df[year_cat_name] = df[year_cat_name].fillna(1828)

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
colors_list = ["cyan", "yellow", "pink", "blue", "green", "yellow", "cyan", "cyan", "cyan", "cyan"] # for genres
media_colors_list = ["purple", "magenta","pink", "grey",  "cyan", "red", "green", "yellow", "orange", "blue", "magenta", "black",
                     ]
colors_list = ["red", "green", "cyan", "blue", "orange" ]
genres_dict = dict(zip(genres_list, colors_list[:len(genres_list)]))

media_dict = dict(zip(media_list, media_colors_list[:len(media_list)]))
media_mpatches_list = []
for genre, color in media_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    media_mpatches_list.append(patch)

media_colors_list = [media_dict[x] for x in media_df[media_cat].values.tolist()]

authors_dict = dict(zip(other_cat_labels_list, colors_list[:len(other_cat_labels_list)]))

genres_mpatches_list = []

for genre, color in genres_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    genres_mpatches_list.append(patch)


genre_colors_list = [genres_dict[x] for x in df[genre_cat_name].values.tolist()]

authors_mpatches_list = []

for genre, color in authors_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    authors_mpatches_list.append(patch)

#authors_colors_list = [authors_dict[x] for x in df["author"].values.tolist()]

serial_mpatches_list = []
for serial, color in {"seriell publiziert":"orange", "nicht-seriell publiziert":"blue"}.items():
    patch = mpatches.Patch(color=color, label=serial)
    serial_mpatches_list.append(patch)

y_variable = "max_value" #"Angstempfinden" #  "lin_susp_model"
x_variables = [year_cat_name ] # "Liebe", "UnbekannteEindr"

if y_variable == "max_value":
    y_variable_legend = "Gefahrenlevel im Text"
elif y_variable == "lin_susp_model":
    y_variable_legend = "Baseline Modell: Gefahr-Angst-Spannung"
else: y_variable_legend = y_variable

fig, axes = plt.subplots(1,2, figsize=(12,8))

axes[0].scatter(media_df["max_value"], media_df["Angstempfinden"], c=media_colors_list)
x = media_df.loc[:, "max_value"]
res = siegelslopes(media_df.loc[:, "Angstempfinden"], x)
axes[0].plot(x, res[1] + res[0] * x, color="grey", linewidth=3)
annotation = media_df.loc["00085-00", "title"]
x_results = media_df.loc["00085-00", "max_value"]
y_results = media_df.loc["00085-00", "Angstempfinden"]
axes[0].annotate(annotation, (x_results, y_results), arrowprops=dict(facecolor='black', shrink=0.05))
id = "00016-00"
annotation = df.loc[id, "title"]
x_results = df.loc[id, "max_value"]
y_results = df.loc[id, "Angstempfinden"]
axes[0].annotate(annotation, (x_results, y_results), arrowprops=dict(facecolor='black', shrink=0.05))
id = "00217-00"
annotation = df.loc[id, "title"]
x_results = df.loc[id, "max_value"]
y_results = df.loc[id, "Angstempfinden"]
axes[0].annotate(annotation, (x_results, y_results), arrowprops=dict(facecolor='black', shrink=0.05))
id = "00310-00"
annotation = df.loc[id, "title"]
x_results = df.loc[id, "max_value"]
y_results = df.loc[id, "Angstempfinden"]
axes[0].annotate(annotation, (x_results, y_results), arrowprops=dict(facecolor='black', shrink=0.05))
axes[0].set_title("Vor 1850")
axes[0].legend(handles=media_mpatches_list)  # authors_mpatches_list
#axes[0].set_ylabel("Angstempfinden")
#plt.xlabel("Gefahrenlevel")
axes[0].set_xlim(0,0.8)
axes[0].set_ylim(0,0.5)






media_list = ["Taschenbuch", "Pantheon"]
media_list = ["Familienblatt", "Rundschau"]

media_df = df[df.isin({media_cat: media_list}).any(axis=1)]
df = df[df.isin({genre_cat_name: genres_list}).any(axis=1)]
df_serial = df[df.isin({"Serialität": ["Serie"]}).any(axis=1)]
df_nonserial = df[df.isin({"Serialität": ["nicht-seriell"]}).any(axis=1)]
colors_list = ["cyan", "yellow", "pink", "blue", "green", "yellow", "cyan", "cyan", "cyan", "cyan"] # for genres
media_colors_list = ["lightgreen", "grey",  "cyan", "red", "green", "yellow", "orange", "blue", "magenta", "black","darkgreen", "lightblue",
                     ]
colors_list = ["red", "green", "cyan", "blue", "orange" ]
genres_dict = dict(zip(genres_list, colors_list[:len(genres_list)]))

media_dict = dict(zip(media_list, media_colors_list[:len(media_list)]))
media_mpatches_list = []
for genre, color in media_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    media_mpatches_list.append(patch)

media_colors_list = [media_dict[x] for x in media_df[media_cat].values.tolist()]

authors_dict = dict(zip(other_cat_labels_list, colors_list[:len(other_cat_labels_list)]))

genres_mpatches_list = []

for genre, color in genres_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    genres_mpatches_list.append(patch)


genre_colors_list = [genres_dict[x] for x in df[genre_cat_name].values.tolist()]

authors_mpatches_list = []

for genre, color in authors_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    authors_mpatches_list.append(patch)

#authors_colors_list = [authors_dict[x] for x in df["author"].values.tolist()]

serial_mpatches_list = []
for serial, color in {"seriell publiziert":"orange", "nicht-seriell publiziert":"blue"}.items():
    patch = mpatches.Patch(color=color, label=serial)
    serial_mpatches_list.append(patch)

y_variable = "max_value" #"Angstempfinden" #  "lin_susp_model"
x_variables = [year_cat_name ] # "Liebe", "UnbekannteEindr"

if y_variable == "max_value":
    y_variable_legend = "Gefahrenlevel im Text"
elif y_variable == "lin_susp_model":
    y_variable_legend = "Baseline Modell: Gefahr-Angst-Spannung"
else: y_variable_legend = y_variable


axes[1].scatter(media_df["max_value"], media_df["Angstempfinden"], c=media_colors_list)
x = media_df.loc[:, "max_value"]
res = siegelslopes(media_df.loc[:, "Angstempfinden"], x)
axes[1].plot(x, res[1] + res[0] * x, color="grey", linewidth=3)
axes[1].set_title("Nach 1850")

id = "00246-00"
annotation = df.loc[id, "title"]
x_results = df.loc[id, "max_value"]
y_results = df.loc[id, "Angstempfinden"]
axes[1].annotate(annotation, (x_results, y_results), arrowprops=dict(facecolor='black', shrink=0.05))

id = "00265-00"
annotation = df.loc[id, "title"]
x_results = df.loc[id, "max_value"]
y_results = df.loc[id, "Angstempfinden"]
axes[1].annotate(annotation, (x_results, y_results), arrowprops=dict(facecolor='black', shrink=0.05))

axes[1].legend(handles=media_mpatches_list)  # authors_mpatches_list

#plt.xlabel("Gefahrenlevel")
axes[1].set_xlim(0,0.8)
axes[1].set_ylim(0,0.5)
fig.suptitle("Korrelation von Gefahr und Angst in Medienformaten")
fig.supxlabel("Gefahrenlevel")
fig.supylabel("Angstempfinden")
plt.tight_layout()
fig.savefig("/home/julian/Documents/CLS_temp/figures/Gefahr_Angst_in_Medienformaten.svg")
plt.show()



print("Finished!")