system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# from my own modules:
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods

# standard libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import cov
from scipy.stats import pearsonr, spearmanr, siegelslopes
import seaborn as sns

medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"

filepath = os.path.join(global_corpus_representation_directory(system), "speech_rep_Matrix.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

matrix_obj = DocFeatureMatrix(data_matrix_filepath=filepath, metadata_csv_filepath= metadata_filepath)
matrix_obj = matrix_obj.add_metadata([genre_cat, year_cat, medium_cat, "Nachname", "Titel",
                                      "in_Deutscher_Novellenschatz", "Kanon_Status", "Gender"])

df1 = matrix_obj.data_matrix_df


length_infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix_obj.data_matrix_df,
                              metadata_csv_filepath=length_infile_df_path)
matrix_obj = matrix_obj.add_metadata(["token_count"])



cat_labels = ["R"]


cat_labels = ["N","E"]
cat_labels = ["N", "E", "0E", "XE", "M", "R"]
cat_labels = ["N", "E", "0E", "XE"]

matrix_obj = matrix_obj.reduce_to_categories(genre_cat, cat_labels)

matrix_obj = matrix_obj.eliminate(["Figuren"])

df = matrix_obj.data_matrix_df

df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1750, end_year=1970, epoch_length=100,
                      new_periods_column_name="periods")

df_before_genre_norm = df.copy()
replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "MLP",
                                    "R": "Roman", "M": "Märchen", "XE": "MLP"}}
df = full_genre_labels(df, replace_dict=replace_dict)


df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {medium_cat: {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)


colors_list = ["red", "green", "cyan","orange", "blue"]
genres_list = df[genre_cat].values.tolist()
list_targetlabels = ", ".join(map(str, set(genres_list))).split(", ")

zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))


zipped_dict = {"Novelle":"red", "Erzählung":"green", "MLP": "cyan", "Märchen":"orange", "Roman":"blue"}
list_colors_target = [zipped_dict[item] for item in genres_list]

df.rename(columns={"Kanon_Status":"Canonicity"}, inplace=True)
canon_cat = "Canonicity"

replace_dict = {canon_cat: {0: "low", 1: "low", 2: "high",
                                                      3:"high"}}
df  = full_genre_labels(df, replace_dict=replace_dict)


mpatches_list = []

for key, value in zipped_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)


novsch_dict = {True:"red", False:"grey"}

replace_dict = {"Gender": {"f": "female", "unbekannt":"unknown", "m": "male"}}
df = full_genre_labels(df, replace_dict=replace_dict)

gender_dict = {"female":"red", "male":"blue", "unknown":"grey"}


canon_dict = {"high":"purple", "low":"grey"}

fig, axes = plt.subplots(3,2, figsize=(12,18))

sns.lineplot(data=df, x="Jahr_ED", y="fraction_dirspeech", hue=canon_cat,
             palette=canon_dict, ax=axes[1,0])
axes[1,0].set_title("Canonicity")
axes[1,0].set_ylabel("Proportion: Direct Speech")
axes[1,0].set_xlabel("")



sns.lineplot(data=df, x="Jahr_ED", y="fraction_indirspeech", hue=canon_cat,
             palette=canon_dict, ax=axes[2,0])
axes[2,0].set_title("Canonicity")
axes[2,0].set_ylabel("Proportion: Indirect Speech")
axes[2,0].set_xlabel("")

sns.lineplot(data=df, x="Jahr_ED", y="fraction_fid", hue=canon_cat,
             palette=canon_dict, ax=axes[0,1])
axes[0,1].set_title("Canonicity – Free Indirect Discourse (FID)")
axes[0,1].set_ylabel("Proportion: FID")
axes[0,1].set_xlabel("")



dirspeech_df = df.loc[:,["Jahr_ED","fraction_dirspeech"]]
dirspeech_df["speech_type"] = "Direct Speech"
dirspeech_df.rename(columns={"fraction_dirspeech":"value"}, inplace=True)

indirspeech_df = df.loc[:,["Jahr_ED","fraction_indirspeech"]]
indirspeech_df["speech_type"] = "Indirect Speech"
indirspeech_df.rename(columns={"fraction_indirspeech":"value"}, inplace=True)

repspeech_df = df.loc[:,["Jahr_ED","fraction_repspeech"]]
repspeech_df["speech_type"] = "FID"
repspeech_df.rename(columns={"fraction_repspeech":"value"}, inplace=True)

fid_df = df.loc[:,["Jahr_ED","fraction_fid"]]
fid_df["speech_type"] = "Reported Speech"
fid_df.rename(columns={"fraction_fid":"value"}, inplace=True)

new_df = pd.concat([dirspeech_df, indirspeech_df, repspeech_df, fid_df])

x = df.loc[:, "Jahr_ED"]
res = siegelslopes(dirspeech_df.loc[:, "value"], df.loc[:,"Jahr_ED"])
axes[0,0].plot(x, res[1] + res[0] * x, color="magenta", linewidth=1)
res = siegelslopes(indirspeech_df.loc[:, "value"], df.loc[:,"Jahr_ED"])
axes[0,0].plot(x, res[1] + res[0] * x, color="orange", linewidth=1)
res = siegelslopes(repspeech_df.loc[:, "value"], df.loc[:,"Jahr_ED"])
axes[0,0].plot(x, res[1] + res[0] * x, color="green", linewidth=1)
res = siegelslopes(fid_df.loc[:, "value"], df.loc[:,"Jahr_ED"])
axes[0,0].plot(x, res[1] + res[0] * x, color="lightblue", linewidth=1)
sns.lineplot(data=new_df, x="Jahr_ED", y="value", hue="speech_type", palette=["magenta", "orange", "green", "lightblue"],
             ax=axes[0,0])
axes[0,0].set_xlabel("")

axes[0,0].set_title("Timeline of Speech Representation")
axes[0,0].set_ylabel("Proportion")


sns.lineplot(data=df, x="Jahr_ED", y="fraction_dirspeech", hue="Gender",
             palette=gender_dict, ax=axes[1,1])
axes[1,1].set_title("Gender")
axes[1,1].set_ylabel("")

sns.lineplot(data=df, x="Jahr_ED", y="fraction_indirspeech", hue="Gender",
             palette=gender_dict, ax=axes[2,1])
axes[2,1].set_title("Gender")
axes[2,1].set_ylabel("")

fig.supxlabel("Year of first publication")
fig.suptitle("Timeline and Canonization for Types of Speech Representation")
fig.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures", "types_speech-rep_canonization.svg"))
plt.show()