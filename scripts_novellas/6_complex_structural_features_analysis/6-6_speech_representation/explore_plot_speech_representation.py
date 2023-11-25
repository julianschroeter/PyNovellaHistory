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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import pymc as pm
from pymc import HalfCauchy, Model, Normal
import arviz as az
import bambi as bmb
from scipy.stats import chi2_contingency
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
                                      "in_Deutscher_Novellenschatz", "Kanon_Status"])

df1 = matrix_obj.data_matrix_df


length_infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix_obj.data_matrix_df,
                              metadata_csv_filepath=length_infile_df_path)
matrix_obj = matrix_obj.add_metadata(["token_count"])



cat_labels = ["R"]

cat_labels = ["N", "E", "0E", "XE", "M", "R"]
cat_labels = ["N","E"]
cat_labels = ["N", "E", "0E", "XE"]

matrix_obj = matrix_obj.reduce_to_categories(genre_cat, cat_labels)

matrix_obj = matrix_obj.eliminate(["Figuren"])

df = matrix_obj.data_matrix_df


print("Covariance. ", cov(df["fraction_dirspeech"], df["fraction_indirspeech"]))

corr, _ = pearsonr(df["fraction_dirspeech"], df["fraction_indirspeech"])
print('Pearsons correlation: %.3f' % corr)

corr, _ = spearmanr(df["fraction_dirspeech"], df["fraction_indirspeech"])
print('Spearman correlation: %.3f' % corr)

df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1750, end_year=1970, epoch_length=100,
                      new_periods_column_name="periods")

df_before_genre_norm = df.copy()
replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "MLP",
                                    "R": "Roman", "M": "Märchen", "XE": "MLP"}}
df = full_genre_labels(df, replace_dict=replace_dict)




#df = df[df.isin({medium_cat:["Familienblatt", "Anthologie", "Taschenbuch", "Rundschau"]}).any(1)]
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


mpatches_list = []

for key, value in zipped_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)

plt.scatter(df['fraction_indirspeech'], df['token_count'], color=list_colors_target, alpha=0.5)
plt.title("Centralization auf Textumfang")
plt.xlabel("indirekte Rede")
plt.ylabel("Textumfang")
plt.legend(handles=mpatches_list)

plt.show()


plt.scatter(df['fraction_dirspeech'], df['token_count'], color=list_colors_target, alpha=0.5)
plt.title("Direkte Rede – Umfang")
plt.xlabel("Anteil direkte Rede")
plt.ylabel("Textumfang")
plt.legend(handles=mpatches_list)

plt.show()

plt.scatter(df["Jahr_ED"], df["fraction_dirspeech"], color=list_colors_target)

x = df.loc[:, "Jahr_ED"]
res = siegelslopes(df.loc[:, "fraction_dirspeech"], x)
print(res)
plt.plot(x, res[1] + res[0] * x, color="black", linewidth=3)


plt.title("Anteil direkte Rede nach Erscheinungsjahr für Gattungen")
plt.legend(handles=mpatches_list)
plt.xlabel("Erstdruck")
plt.ylabel("Anteil direkte Rede")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="fraction_dirspeech", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)
plt.plot(x, res[1] + res[0] * x, color="black", linewidth=3)
plt.title("Gattungen nach Anteil direkter Rede")
plt.ylabel("Anteil direkte Rede")
plt.show()


sns.lineplot(data=df, x="Jahr_ED", y="fraction_fid", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)
plt.title("Gattungen nach Anteil FID")
plt.show()


sns.lineplot(data=df, x="Jahr_ED", y="fraction_indirspeech", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)
plt.title("Gattungen nach Anteil indirekter Rede")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="fraction_repspeech", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)
plt.title("Gattungen nach Anteil erzählter Rede")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="fraction_dirspeech", hue=medium_cat,
             )
plt.title("Mediensorten nach Anteil direkter Rede")
plt.show()




novsch_dict = {True:"red", False:"grey"}

sns.lineplot(data=df, x="Jahr_ED", y="fraction_indirspeech", hue="in_Deutscher_Novellenschatz",
             palette=novsch_dict)
plt.title("Novellenschatz – Redewiedergabetyp")
plt.show()

replace_dict = {"Kanon_Status": {0: "niedrig", 1: "niedrig", 2: "hoch",
                                                  3: "hoch"}}
df = full_genre_labels(df, replace_dict=replace_dict)
canon_dict = {"hoch":"red", "niedrig":"grey"}

sns.lineplot(data=df, x="Jahr_ED", y="fraction_indirspeech", hue="Kanon_Status",
             palette=canon_dict)
plt.title("Gattungen nach Anteil des Redewiedergabetyps")
plt.ylabel("Anteil indirekte Rede")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="fraction_dirspeech", hue="Kanon_Status",
             palette=canon_dict)
plt.title("Gattungen nach Anteil des Redewiedergabetyps")
plt.ylabel("Anteil direkte Rede")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="fraction_fid", hue="Kanon_Status",
             palette=canon_dict)
plt.title("Gattungen nach Anteil des Redewiedergabetyps")
plt.ylabel("Anteil erlebte Rede")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="fraction_repspeech", hue="Kanon_Status",
             palette=canon_dict)
plt.title("Gattungen nach Anteil des Redewiedergabetyps")
plt.ylabel("Anteil erzählte Rede")
plt.show()

mpatches_list = []

for key, value in canon_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)

data = df.copy()
periods = list(set(data.periods.values.tolist()))
for period in periods:
    df = data[data["periods"] == period]
    canon_list = df["Kanon_Status"].values.tolist()
    list_colors_target = [canon_dict[item] for item in canon_list]

    plt.scatter(df['fraction_indirspeech'], df['fraction_dirspeech'], color=list_colors_target, alpha=0.5)
    plt.title("Indirekte Rede auf direkte Rede: " + str(period))
    plt.xlabel("Indirekte Rede")
    plt.ylabel("Direkte Rede")
    plt.ylim(0,1)
    plt.xlim(0,0.3)
    plt.legend(handles=mpatches_list)
    plt.show()


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "MLP", "E": "MLP", "0E": "MLP",
                                    "R": "Roman", "M": "Märchen", "XE": "MLP"}}
df = df_before_genre_norm.copy()
df = full_genre_labels(df, replace_dict=replace_dict)
sns.lineplot(data=df, x="Jahr_ED", y="fraction_dirspeech", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)
plt.title("Gattungen nach Anteil direkter Rede")
plt.show()


sns.lineplot(data=df, x="Jahr_ED", y="fraction_fid", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)
plt.title("Gattungen nach Anteil FID")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="fraction_repspeech", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)
plt.title("Gattungen nach Anteil erzählte Rede")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="fraction_indirspeech", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)
plt.title("Gattungen nach Anteil indirekte Rede")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="fraction_dirspeech", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)
plt.title("Gattungen nach Anteil indirekte Rede")
plt.show()

dirspeech_df = df.loc[:,["Jahr_ED","fraction_dirspeech"]]
dirspeech_df["speech_type"] = "Direkte Rede"
dirspeech_df.rename(columns={"fraction_dirspeech":"value"}, inplace=True)

indirspeech_df = df.loc[:,["Jahr_ED","fraction_indirspeech"]]
indirspeech_df["speech_type"] = "Indir. Rede"
indirspeech_df.rename(columns={"fraction_indirspeech":"value"}, inplace=True)

repspeech_df = df.loc[:,["Jahr_ED","fraction_repspeech"]]
repspeech_df["speech_type"] = "Erzählte Rede"
repspeech_df.rename(columns={"fraction_repspeech":"value"}, inplace=True)

fid_df = df.loc[:,["Jahr_ED","fraction_fid"]]
fid_df["speech_type"] = "Erlebte Rede"
fid_df.rename(columns={"fraction_fid":"value"}, inplace=True)

new_df = pd.concat([dirspeech_df, indirspeech_df, repspeech_df, fid_df])

x = df.loc[:, "Jahr_ED"]
res = siegelslopes(dirspeech_df.loc[:, "value"], df.loc[:,"Jahr_ED"])
plt.plot(x, res[1] + res[0] * x, color="blue", linewidth=1)
res = siegelslopes(indirspeech_df.loc[:, "value"], df.loc[:,"Jahr_ED"])
plt.plot(x, res[1] + res[0] * x, color="orange", linewidth=1)
res = siegelslopes(repspeech_df.loc[:, "value"], df.loc[:,"Jahr_ED"])
plt.plot(x, res[1] + res[0] * x, color="green", linewidth=1)
res = siegelslopes(fid_df.loc[:, "value"], df.loc[:,"Jahr_ED"])
plt.plot(x, res[1] + res[0] * x, color="red", linewidth=1)
sns.lineplot(data=new_df, x="Jahr_ED", y="value", hue="speech_type", palette=["blue", "orange", "green", "red"])
plt.title("Tendenz der Redewiedergabetypen (MLP)")
plt.ylabel("Wert: Redewiedergabetyp")
plt.xlabel("Jahr des Erstdrucks")
plt.ylim(0,1)
plt.savefig(os.path.join(local_temp_directory(system), "figures", "Tendenz_Redewiedergabetypen.svg"))
plt.show()