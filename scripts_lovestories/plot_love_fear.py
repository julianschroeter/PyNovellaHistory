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
from scipy.stats import pearsonr, spearmanr, siegelslopes, ttest_ind, f_oneway
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np

year_cat_name = "Erscheinungsjahr"
genre_cat_name = "Gattungen"
media_cat = "Medium"



novellas_infilepath = os.path.join(local_temp_directory(system),  "AllChunksDangerFearCharacters_novellas_episodes_scaled.csv")
novellas_infilepath = os.path.join(local_temp_directory(system), "AllChunksDangerFearCharactersHardseeds_novellas_episodes_scaled.csv")
novellas_infilepath = os.path.join(local_temp_directory(system), "MaxDangerFearCharactersHardseeds_novellas_Ganztexte_scaled.csv")
novellas_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")



novellas_df = pd.read_csv(novellas_infilepath, index_col = 0)
novellas_dtm_obj = DocFeatureMatrix(data_matrix_filepath=novellas_infilepath, metadata_csv_filepath=novellas_metadata_filepath)
novellas_dtm_obj = novellas_dtm_obj.add_metadata(["Titel", "Nachname","Jahr_ED","Gattungslabel_ED_normalisiert","Medium_ED",
                                                  "Kanon_Status", "seriell", "in_Deutscher_Novellenschatz",
                                                  "Gender"])
novellas_df = novellas_dtm_obj.data_matrix_df
novellas_df_full_media = standardize_meta_data_medium(df=novellas_df, medium_column_name="Medium_ED")

novellas_df = novellas_df_full_media.drop(columns=["Medium_ED", "medium"])

novellas_df = novellas_df.rename(columns={"Titel": "title", "Nachname":"author", "Jahr_ED":"Erscheinungsjahr",
                                          "Gattungslabel_ED_normalisiert":"Gattungen", "medium_type": "Medium",
                                          "seriell":"Serialität", "Kanon_Status":"Canonicity"})

df = novellas_df

df = df[df["Angstempfinden"].notna()]

replace_dict = {"Canonicity": {0: "low", 1: "low", 2: "high",
                                                  3:"high"}}
df = full_genre_labels(df, replace_dict=replace_dict)

replace_dict = {"Gender": {"f": "female", "unbekannt":"unknown", "m": "male"}}
df = full_genre_labels(df, replace_dict=replace_dict)

labels_list = ["R", "M", "E", "N", "0E", "XE", "0P", "0PB", "krimi", "abenteuer", "krieg"]
labels_list = ["R", "E", "N", "0E", "XE", "M"]
labels_list = ["E", "N", "0E", "XE", "M"]
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

df[year_cat_name] = df[year_cat_name].fillna(1828)

genres_list = ["N", "E", "R", "0E", "XE", "M", "0P", "0PA", "0X_Essay", "0PB"]

genres_list = ["MLP", "Märchen"] #, "Spannungs-Heftroman", "Roman"
genres_list = ["Novelle", "Erzählung", "sonst. MLP", "Roman", "Märchen"]

media_list = ["Familienblatt", "Rundschau"]
media_list = ["Taschenbuch", "Pantheon"]
media_list = ["Pantheon", "Journal", "Taschenbuch", "Familienblatt", "Rundschau", "Anthologie"]

media_df = df[df.isin({media_cat: media_list}).any(axis=1)]

gender_list = ["male", "female"]
df_gender = df[df.isin({"Gender": gender_list}).any(axis=1)]
df_male = df_gender[df_gender["Gender"] == "male"]
df_female = df_gender[df_gender["Gender"] == "female"]

df = df[df.isin({genre_cat_name: genres_list}).any(axis=1)]
df_serial = df[df.isin({"Serialität": ["Serie"]}).any(axis=1)]
df_nonserial = df[df.isin({"Serialität": ["nicht-seriell"]}).any(axis=1)]

df_canon_high = df[df.isin({"Canonicity": ["high"]}).any(axis=1)]
df_canon_low = df[df.isin({"Canonicity": ["low"]}).any(axis=1)]

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
for serial, color in {"seriell publiziert":"orange", "nicht-seriell publiziert":"grey"}.items():
    patch = mpatches.Patch(color=color, label=serial)
    serial_mpatches_list.append(patch)

gender_color_dict = {"male":"blue", "female":"red", "unknown":"grey"}
gender_mpatches_list = []
for serial, color in gender_color_dict.items():
    patch = mpatches.Patch(color=color, label=serial)
    gender_mpatches_list.append(patch)

gender_colors_list = [gender_color_dict[x] for x in df_gender["Gender"].values.tolist()]


# ANOVA for gender and love for the feature "love":

F, p = f_oneway(df_male["Liebe"], df_female["Liebe"])
print("F, p statistics of ANOVA test for male versus female for Love:", F, p)

F, p = f_oneway(df_canon_high["Liebe"], df_canon_low["Liebe"])
print("F, p statistics of ANOVA test for canon high versus canon low for Love:", F, p)




y_variable = "Farben" #"Abstrakta" # "Hardseeds" # "fear_love" #"Liebe" # "max_value" #"Angstempfinden" #  "lin_susp_model"
x_variables = [year_cat_name ] # , "UnbekannteEindr"

if y_variable == "max_value":
    y_variable_legend = "Gefahrenlevel im Text"
elif y_variable == "lin_susp_model":
    y_variable_legend = "Baseline Modell: Gefahr-Angst-Spannung"
else: y_variable_legend = y_variable

sns.lineplot(data=df_gender, x=year_cat_name, y=y_variable, hue="Gender")
plt.title("Zeitlicher Verlauf des Gefahrenlevels")
plt.ylabel(y_variable_legend)
plt.xlabel("Jahr des Erstdrucks")
plt.show()

sns.lineplot(data=df_gender, x=year_cat_name, y="Angstempfinden", hue="Gender")
plt.title("Zeitlicher Verlauf der Figurenangst")
plt.ylabel("Angstempfinden der Figur")
plt.xlabel("Jahr des Erstdrucks")
plt.show()

sns.lineplot(data=df_gender, x=year_cat_name, y="max_value", hue="Gender")
plt.title("Zeitlicher Verlauf der Figurenangst")
plt.ylabel("Gefahrenlevel im Text")
plt.xlabel("Jahr des Erstdrucks")
plt.show()

sns.lineplot(data=df_gender, x=year_cat_name, y="Liebe", hue="Gender")
plt.title("Zeitlicher Verlauf der Figurenangst")
plt.ylabel("Liebe im Text")
plt.xlabel("Jahr des Erstdrucks")
plt.show()


fig, axes = plt.subplots(1,2, figsize=(12,6))
sns.lineplot(data=df, x=year_cat_name, y="Liebe", hue="Canonicity",
             palette={"low":"grey", "high":"purple"}, ax= axes[0])
axes[0].set_title("Canonization")
axes[0].set_ylabel("Love vocabulary")
axes[0].set_ylim(0,0.6)
axes[0].set_xlabel("")

text_id = "00047-00"
annotation = df.loc[text_id, "title"]
print(annotation)
x_results = df.loc[text_id, year_cat_name]
y_results = df.loc[text_id, "Liebe"]
axes[0].annotate(annotation, (x_results, y_results), arrowprops=dict(facecolor='black', shrink=0.05))

text_id = "00306-00"
annotation = df.loc[text_id, "title"]
print(annotation)
x_results = df.loc[text_id, year_cat_name]
y_results = df.loc[text_id, "Liebe"]
axes[0].annotate(annotation, (x_results, y_results), arrowprops=dict(facecolor='black', shrink=0.05))

fig.supxlabel("Year of first publication")

sns.lineplot(data=df, x=year_cat_name, y="Liebe", hue="Gender",
             palette=gender_color_dict, ax=axes[1])
axes[1].set_title("Gender")
axes[1].set_ylabel("")
axes[1].set_ylim(0,0.6)
axes[1].set_xlabel("")
fig.suptitle("Love in Canon and Gender")
fig.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system), "figures", "Figure1_Lineplots_love-canon-gender-timeline.svg"))
fig.show()


plt.scatter(df_gender["Liebe"], df_gender["Angstempfinden"], c=gender_colors_list, alpha=0.5)
x = df_male.loc[:, "Liebe"]
res = siegelslopes(df_male.loc[:, "Angstempfinden"], x)
print("male: ", x, res)
plt.plot(x, res[1] + res[0] * x, color="black", linewidth=5)
x = df_female.loc[:, "Liebe"]
res = siegelslopes(df_female.loc[:, "Angstempfinden"], x)
print("female: ", x, res)
plt.plot(x, res[1] + res[0] * x, color="red", linewidth=3)
plt.legend(handles=gender_mpatches_list)  # authors_mpatches_list
plt.ylabel("Angstempfinden")
plt.xlabel("Liebe")
plt.title("Korrelation von Liebe und Figurenangst nach Gender")
#plt.xlim(0,0.5)
#plt.ylim(0,0.5)
plt.show()

whole_df = df.copy()
whole_df.boxplot(column=y_variable, by="Gender")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column=y_variable, by="Canonicity")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="Angstempfinden", by="Gender")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

media_df.boxplot(column="Liebe", by="Gender")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="max_value", by="Gender")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


whole_df.boxplot(column="Liebe", by="Canonicity")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="Angstempfinden", by="Canonicity")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="max_value", by="Canonicity")
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()




print("Finished!")