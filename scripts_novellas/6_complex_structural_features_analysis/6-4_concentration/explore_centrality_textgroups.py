system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# from my own modules:
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods
from preprocessing.sna import get_centralization, scale_centrality
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

filename = "conll_based_networkdata-matrix-novellas.csv"
filepath = os.path.join(local_temp_directory(system), filename)
#filepath = os.path.join(global_corpus_representation_directory(system), "Network_Matrix_all.csv")

metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


matrix_obj = DocFeatureMatrix(data_matrix_filepath=filepath, metadata_csv_filepath= metadata_filepath)
df1 = matrix_obj.data_matrix_df

matrix_obj = matrix_obj.add_metadata([genre_cat, year_cat, medium_cat, "Nachname", "Titel", "Kanon_Status", "in_Deutscher_Novellenschatz"])



length_infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix_obj.data_matrix_df,
                              metadata_csv_filepath=length_infile_df_path)
matrix_obj = matrix_obj.add_metadata(["token_count"])

cat_labels = ["N", "E"]
cat_labels = ["0E", "XE"]
cat_labels = ["N"]
cat_labels = ["E"]
cat_labels = ["R"]
cat_labels = ["N", "E", "0E", "XE"]
cat_labels = ["N", "E", "0E", "XE", "M", "R"]
matrix_obj = matrix_obj.reduce_to_categories(genre_cat, cat_labels)

matrix_obj = matrix_obj.eliminate(["Figuren"])

df = matrix_obj.data_matrix_df

print(df)

df = df[~df["token_count"].isna()]
#df = df[~df["Netzwerkdichte_rule"].isna()]

df = df[df["Figurenanzahl"]>2]
df = df[df["degree_centrality"] != "{'NO_CHARACTER': 0}"]
df["scaled_centrality_conll"] = df.apply(lambda x: scale_centrality(eval(x["degree_centrality"]), dict(eval(x["weighted_deg_centr"]))), axis=1)
df["scaled_centralization_conll"] = df.apply(lambda x: get_centralization(dict(x["scaled_centrality_conll"]), c_type="degree"), axis=1)


df.rename(columns={"scaled_centralization_conll": "Zentralisierung"}, inplace=True)
df.rename(columns={"density": "Netzwerkdichte"}, inplace=True)
#df.rename(columns={"Figurenanzahl_conll": "Figurenanzahl"}, inplace=True)


df = df[~df["Zentralisierung"].isna()]


print(df["Zentralisierung"].dtypes)
df["Zentralisierung"] = df["Zentralisierung"].astype(float)





print("Zentralisierung: Covariance. ", cov(df["Zentralisierung"], df["token_count"]))
corr, _ = pearsonr(df["Zentralisierung"], df["token_count"])
print('Zentralisierung: Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(df["Zentralisierung"], df["token_count"])
print('Zentralisierung: Spearman correlation: %.3f' % corr)


print("Netzwerkdichte: Covariance. ", cov(df["Netzwerkdichte"], df["token_count"]))
corr, _ = pearsonr(df["Netzwerkdichte"], df["token_count"])
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(df["Netzwerkdichte"], df["token_count"])
print('Spearman correlation: %.3f' % corr)


print("Zentralisierung und Figurenanzahl: Covariance. ", cov(df["Zentralisierung"], df["Figurenanzahl"]))
corr, _ = pearsonr(df["Zentralisierung"], df["token_count"])
print('Zentralisierung und Figurenanzahl: Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(df["Zentralisierung"], df["Figurenanzahl"])
print('Spearman correlation: %.3f' % corr)


print("Netzwerkdichte und Figurenanzahl: Covariance. ", cov(df["Netzwerkdichte"], df["Figurenanzahl"]))
corr, _ = pearsonr(df["Netzwerkdichte"], df["Figurenanzahl"])
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(df["Netzwerkdichte"], df["Figurenanzahl"])
print('Spearman correlation: %.3f' % corr)


print("Netzwerkdichte  - Zentralisierung: Covariance. ", cov(df["Netzwerkdichte"], df["Zentralisierung"]))
corr, _ = pearsonr(df["Netzwerkdichte"], df["Zentralisierung"])
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(df["Netzwerkdichte"], df["Zentralisierung"])
print('Spearman correlation: %.3f' % corr)


df = df[df["Jahr_ED"] >= 1800]
df = df[df["Jahr_ED"] <=1950]

df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1800, end_year=1950, epoch_length=50,
                      new_periods_column_name="periods")


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "MLP", "E": "MLP", "0E": "MLP",
                                    "R": "Roman", "M": "Märchen", "XE": "MLP"}}
replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "MLP",
                                    "R": "Roman", "M": "Märchen", "XE": "MLP"}}


df = full_genre_labels(df, replace_dict=replace_dict)





df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)

#df = df[df.isin({medium_cat:["Familienblatt", "Anthologie", "Taschenbuch", "Rundschau", "Journal"]}).any(axis=1)]

colors_list = ["red", "green", "cyan","orange", "blue"]
genres_list = df[genre_cat].values.tolist()
list_targetlabels = ", ".join(map(str, set(genres_list))).split(", ")

zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))
zipped_dict = {"MLP": "cyan", "Märchen":"orange", "Roman":"blue"}
zipped_dict = {"Novelle":"red", "Erzählung":"green", "MLP": "cyan", "Märchen":"orange", "Roman":"blue"}
list_colors_target = [zipped_dict[item] for item in genres_list]


mpatches_list = []

for key, value in zipped_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)

media_list = df[medium_cat].values.tolist()
media_dict = {"Journal":"magenta", "Buch":"brown", "Taschenbuch":"pink", "Anthologie":"grey",
              "Familienblatt":"yellow", "Rundschau":"lightblue"}
media_colors_target = [media_dict[item] for item in media_list]
media_mpatches_list = []
for key, value in media_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    media_mpatches_list.append(patch)

df1 = df.copy()

plt.scatter(df['token_count'], df['Zentralisierung'], color=list_colors_target, alpha=0.5)
plt.title("Zentralisierung - Textumfang")
plt.ylabel("Zentralisierung")
plt.xlabel("Textumfang")
plt.xlim(0,200000)
plt.legend(handles=mpatches_list)
plt.show()


plt.scatter(df['token_count'],df['Netzwerkdichte'],  color=list_colors_target, alpha=0.5)
plt.title("Netzwerkdichte – Textumfang")
plt.ylabel("Netzwerkdichte")
plt.xlabel("Textumfang")
plt.xlim(0,200000)
plt.legend(handles=mpatches_list)
plt.show()


plt.scatter(df['token_count'],df['Figurenanzahl'],  color=list_colors_target, alpha=0.5)
plt.title("Umfang - Figurenanzahl")
plt.ylabel("Figurenanzahl")
plt.xlabel("Umfang")
plt.xlim(0,50000)
plt.ylim(0,50)
plt.legend(handles=mpatches_list)
plt.show()


plt.scatter(df['Figurenanzahl'], df['Zentralisierung'], color=list_colors_target, alpha=0.5)
plt.title("Figurenanzahl und Zentralisierung")
plt.ylabel("Zentralisierung")
plt.xlabel("Figurenanzahl")
plt.xlim(0,100)
plt.legend(handles=mpatches_list)
plt.show()


plt.scatter(df['Figurenanzahl'],df['Netzwerkdichte'],  color=list_colors_target, alpha=0.5)
plt.title("Figurenanzahl und Dichte ")
plt.ylabel("Netzwerkdichte")
plt.xlabel("Figurenanzahl")
plt.xlim(0,100)
plt.legend(handles=mpatches_list)
plt.show()


plt.scatter(df["Zentralisierung"], df["Netzwerkdichte"], color=list_colors_target, alpha=0.5)
plt.title("Zentralisierung und Netzwerkdichte")
plt.legend(handles=mpatches_list)
plt.xlabel("Zentralisierung")
plt.ylabel("Netzwerkdichte")
plt.show()


plt.scatter(df["Jahr_ED"], df["Zentralisierung"], color=list_colors_target)

x = df.loc[:, "Jahr_ED"]
res = siegelslopes(df.loc[:, "Zentralisierung"], x)
plt.plot(x, res[1] + res[0] * x, color="black", linewidth=1)

plt.title("Zentralisierung nach Erscheinungsjahr für Gattungen")
plt.legend(handles=mpatches_list)
plt.xlabel("Erstdruck")
plt.ylabel("Zentralisierung")
plt.show()


plt.scatter(df["Jahr_ED"], df["Zentralisierung"], color=media_colors_target)
plt.title("Zentralisierung nach Erscheinungsjahr für Medienformate")
plt.legend(handles=media_mpatches_list)
plt.xlabel("Erstdruck")
plt.ylabel("Zentralisierung")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="Zentralisierung", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)
plt.plot(x, res[1] + res[0] * x, color="black", linewidth=1)
plt.ylim(0,1)
plt.title("Gattungen nach Grad an Zentralisierung")
plt.show()




#df["Netzwerkdichte"] = df.apply(lambda x: (x["Netzwerkdichte"] + x["Netzwerkdichte_conll"]) / 2, axis=1)
sns.lineplot(data=df, x="Jahr_ED", y="Netzwerkdichte", hue="Gattungslabel_ED_normalisiert",
             palette=zipped_dict)

# palette=["red", "orange", "cyan", "green", "blue"]
plt.title("Gattungen nach Grad an Netzwerkdichte")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="Zentralisierung", hue="Medientyp_ED")
plt.show()


data = df.loc[:, ("Zentralisierung",genre_cat,medium_cat, "Kanon_Status", "periods", "token_count", "Netzwerkdichte", year_cat)]


zipped_dict = {True:"red", False:"grey"}

sns.lineplot(data=df, x="Jahr_ED", y="Zentralisierung", hue="in_Deutscher_Novellenschatz",
             palette=zipped_dict)
plt.title("Zentralisierung – Novellenschatz")
plt.show()


canon_dict = {"hoch":"purple", "niedrig":"cyan"}
replace_dict = {"Kanon_Status": {0: "niedrig", 1: "niedrig", 2: "hoch",
                                                  3: "hoch"}}
data = full_genre_labels(data, replace_dict=replace_dict)


#sns.lineplot(data=df, x="Jahr_ED", y="Zentralisierung", hue="Kanon_Status",
 #            palette=zipped_dict)
#plt.title("Gattungen nach Grad an Zentralisierung")
#plt.show()
zipped_dict = {"Novelle":"red", "Erzählung":"green", "MLP": "cyan", "Märchen":"orange", "Roman":"blue"}
mpatches_list = []

for key, value in zipped_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)

canon_mpatches_list = []

for key, value in canon_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    canon_mpatches_list.append(patch)


fig, axes = plt.subplots(3,2, figsize=[8, 12])
periods = list(set(data.periods.values.tolist()))
periods.sort()
print(periods)
i = 0
for period in periods:
    print(period)
    df = data[data["periods"] == period]
    canon_list = df["Kanon_Status"].values.tolist()
    media_period_list = df["Medientyp_ED"].values.tolist()
    list_colors_target = [canon_dict[item] for item in canon_list]

    axes[i,0].scatter(df['token_count'], df['Zentralisierung'], color=list_colors_target, alpha=0.5)
    axes[i,0].set_title("Kanon-Status: " + str(period))
    
    axes[i,0].set_ylim(0,1)
    axes[i,0].set_xlim(0,60000)
    i +=1
axes[0,0].legend(handles=canon_mpatches_list)
i = 0
for period in periods:
    df = data[data["periods"] == period]
    genre_list = df[genre_cat].values.tolist()
    list_colors_target = [zipped_dict[item] for item in genre_list]

    axes[i,1].scatter(df['token_count'], df['Zentralisierung'], color=list_colors_target, alpha=0.5)
    axes[i,1].set_title("Gattungen: " + str(period))
    axes[i,1].set_ylim(0,1)
    axes[i,1].set_xlim(0,60000)
    i += 1
axes[0,1].legend(handles=mpatches_list)

fig.supxlabel("Textumfang")
fig.supylabel("Zentralisierung")
fig.suptitle("Umfang – Zentralisierung in Gattung und Kanon")
fig.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system), "figures", "Umfang_Zentralisierung_Subplots_Gattungen_Kanon_Zeitabschnitt.svg"))
plt.show()

genres = list(set(data.Gattungslabel_ED_normalisiert.values.tolist()))
for genre in genres:
    df = data[data["Gattungslabel_ED_normalisiert"] == genre]
    #canon_list = df["Kanon_Status"].values.tolist()
    media_period_list = df["Medientyp_ED"].values.tolist()
    list_colors_target = [media_dict[item] for item in media_period_list]

    plt.scatter(df['Jahr_ED'], df['Zentralisierung'], color=list_colors_target, alpha=0.5)
    plt.title("Zentralisierung für Medienformate im Genre: " + str(genre))
    plt.xlabel("Jahr Erstdruck")
    plt.ylabel("Zentralisierung")
   # plt.ylim(0,1)
    #plt.xlim(0,1)
    plt.legend(handles=media_mpatches_list)
    plt.show()