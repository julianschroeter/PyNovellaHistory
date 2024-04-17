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
filename = "conll_based_networkdata-matrix-novellas_15mostcommon.csv"
filepath = os.path.join(local_temp_directory(system), filename)
#filepath = os.path.join(global_corpus_representation_directory(system), "Network_Matrix_all.csv")

metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


matrix_obj = DocFeatureMatrix(data_matrix_filepath=filepath, metadata_csv_filepath= metadata_filepath)
df1 = matrix_obj.data_matrix_df

matrix_obj = matrix_obj.add_metadata([genre_cat, year_cat, medium_cat, "Nachname", "Titel", "Kanon_Status"])



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


print("Umfang – Figurenanzahl: Covariance. ", cov(df["token_count"], df["Figurenanzahl"]))
corr, _ = pearsonr(df["token_count"], df["Figurenanzahl"])
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(df["token_count"], df["Figurenanzahl"])
print('Spearman correlation: %.3f' % corr)


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

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,8))

axes[0].scatter(df['token_count'], df['Zentralisierung'], color=list_colors_target, alpha=0.5)
#axes[0].set_title("Zentralisierung - Textumfang")
axes[0].set_ylabel("Zentralisierung")
axes[0].set_xlim(0,200000)
axes[0].set_ylim(0,1)
#plt.legend(handles=mpatches_list)
plt.show()


axes[1].scatter(df['token_count'],df['Netzwerkdichte'],  color=list_colors_target, alpha=0.5)
#axes[1].set_title("Netzwerkdichte – Textumfang")
axes[1].set_ylabel("Netzwerkdichte")
axes[1].set_ylim(0,1)
axes[1].set_xlim(0,200000)
axes[1].legend(handles=mpatches_list)
fig.suptitle("Umfang auf Zentralisierung und Dichte \n15 häufigste Figuren")
fig.supxlabel("Textumfang")
fig.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system),"figures", "15mostcommon_plot_Umfang-auf-Zentralisierung-und-Dichte.svg"))
plt.show()


plt.scatter(df['token_count'],df['Figurenanzahl'],  color=list_colors_target, alpha=0.5)

x = df.loc[:, "token_count"]
res = siegelslopes(df.loc[:, "Figurenanzahl"], x)
plt.plot(x, res[1] + res[0] * x, color="black", linewidth=1)

plt.title("Umfang - Figurenanzahl")
plt.ylabel("Figurenanzahl")
plt.xlabel("Umfang")
plt.xlim(0,200000)
plt.ylim(0,200)
plt.legend(handles=mpatches_list)
plt.show()


sns.regplot(x = "token_count", y = "Figurenanzahl", data= df,
            scatter_kws = {"color": "black", "alpha": 0.5},
            line_kws = {"color": "red"},
            ci = 99) # 99% level
plt.title("Umfang – Figurenanzahl")
plt.ylabel("Figurenanzahl")
plt.xlabel("Umfang")
plt.xlim(0,200000)
plt.ylim(0,200)
plt.savefig(os.path.join(local_temp_directory(system),"figures", "15mc_regplot_Umfang_Figurenanzahl.svg"))
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
plt.savefig(os.path.join(local_temp_directory(system),"figures", "Korrelation_Zentralisierung_Netzwerkdichte.svg"))
plt.show()