system = "my_xps"  # "wcph113" #
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

from preprocessing.presetting import global_corpus_representation_directory, language_model_path, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import years_to_periods, full_genre_labels
import matplotlib.pyplot as plt
import os
from scipy import stats
import pandas as pd
import numpy as np
my_model_de = language_model_path(system)

infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
metadata_df_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_df_path, metadata_csv_filepath=metadata_df_path)
matrix_obj = matrix_obj.reduce_to(["token_count"])
matrix_obj = matrix_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED", "Medientyp_ED", "in_Deutscher_Novellenschatz", "Kanon_Status"])
matrix_obj.data_matrix_df = years_to_periods(matrix_obj.data_matrix_df, category_name="Jahr_ED", start_year=1760, end_year=1970, epoch_length=10, new_periods_column_name="periods")

cat_labels = ["N", "E", "XE"]
matrix_obj = matrix_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)


replace_dict = {"Medientyp_ED": {"Zeitschrift": "Journal", "Zeitung": "Journal",
                                 "Kalender": "Kalender", "Rundschau" : "Rundschau",
                                 "Zyklus" : "Anthologie", "Roman" : "Buch",
                                 "(unbekannt)" : "(unbekannt)",
                                    "Illustrierte": "Journal", "Sammlung": "Anthologie",
                                 "Nachlass": "Buch", "Jahrbuch":"Jahrbuch",
                                 "Monographie": "Buch", "Werke": "Buch"}}
matrix_obj.data_matrix_df = full_genre_labels(matrix_obj.data_matrix_df, replace_dict=replace_dict)



#matrix_obj = matrix_obj.reduce_to_categories("periods", ["1800-1810","1810-1820","1820-1830", "1830-1840", "1840-1850", "1850-1860"])
#matrix_obj = matrix_obj.reduce_to_categories("Medientyp_ED", ["Taschenbuch", "Anthologie", "Buch", "Familienblatt"])

df = matrix_obj.data_matrix_df


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "MLP", "0E": "MLP",
                                    "R": "Roman", "M": "Märchen", "XE": "MLP"}}
df = full_genre_labels(df, replace_dict=replace_dict)


df = df.rename(columns={"in_Deutscher_Novellenschatz": "Novellenschatz"})


replace_dict = {"Novellenschatz": {False: "sonst. MLP", True:"Novellenschatz"}}
df = full_genre_labels(df, replace_dict=replace_dict)


replace_dict = {"Kanon_Status": {3:"Kanon"}}
df = full_genre_labels(df, replace_dict=replace_dict)


#df = df[df["Medientyp_ED"] == "Anthologie"] # reduce to Taschenbücher
df = df[df["Jahr_ED"] >= 1810] # remove all rows where ...
df = df[df["Jahr_ED"] <= 1900] # remove all rows where ...


#df = df[df["Gattungslabel_ED_normalisiert"] != "Roman"] # remove all novels

# ANOVA with scipy.stats: calculte F-statistic and p-value

df_NovSch = df[df["Novellenschatz"]== "Novellenschatz"]
df_Non_NovSch = df[df["Novellenschatz"]== "sonst. MLP"]
df_canon3 = df[df["Kanon_Status"]== "Kanon"]


N_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'Novelle']
E_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'Erzählung']
otherE_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'andere Label']
MLP0_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'kein Label']
all_MLP_df = df[df.isin({"Gattungslabel_ED_normalisiert": ["Novelle", "Erzählung", "MLP0"]}).any(axis=1)]
N_E_df = df[df.isin({"Gattungslabel_ED_normalisiert": ["Novelle", "Erzählung"]}).any(axis=1)]
all_MLP_df = all_MLP_df.loc[:,"token_count"]
N_E_df = N_E_df.loc[:,"token_count"]

R_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'Roman']
M_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'Märchen']

TB_df = df['token_count'][df['Medientyp_ED'] == 'Taschenbuch']
Ant_df = df['token_count'][df['Medientyp_ED'] == 'Anthologie']
FB_df = df['token_count'][df['Medientyp_ED'] == 'Familienblatt']
RS_df = df['token_count'][df['Medientyp_ED'] == 'Rundschau']
Buch_df = df['token_count'][df['Medientyp_ED'] == 'Buch']

import seaborn as sns
fig, ax = plt.subplots()
sns.lineplot(x="Jahr_ED", y="token_count", hue="Gattungslabel_ED_normalisiert", data=df, palette=["red", "cyan", "green"])
sns.lineplot(x="Jahr_ED", y="token_count", hue="Novellenschatz", data=df_NovSch, palette=["orange"])
sns.lineplot(x="Jahr_ED", y="token_count", hue="Kanon_Status", data=df_canon3, palette=["blue"])

plt.ylabel("Textlänge")
plt.xlabel("Jahr")
plt.title("Länge nach Gattungsbezeichnungen, Novellenschatz und Kanon")
plt.ylim(0,100000)
plt.tight_layout()
plt.savefig("/home/julian/Documents/CLS_temp/figures/Abb_Länge_Kanon_Novellenschatz.svg")
plt.show()

fig, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].boxplot([N_df.values.tolist(), E_df.values.tolist(), otherE_df.values.tolist(),
                  M_df.values.tolist()], labels=["›Novelle‹", "›Erzählung‹", "sonst_MLP",  "Märchen"])
axes[0].set_title("Gattungen")
axes[0].set_ylabel("Textlänge in Wort-Token")
left, right = axes[0].set_xlim()
axes[0].hlines(y=50000,xmin=left, xmax=right, color="red")
axes[0].set_ylim([0, 150000])

axes[1].boxplot([TB_df.values.tolist(), Ant_df.values.tolist(), FB_df.values.tolist(), RS_df.values.tolist(), Buch_df.values.tolist()],
                labels=["Taschenbuch", "Anthologie", "Familienblatt", "Rundschau", "Buch"])
axes[1].set_title("Medientypen")
axes[1].set_yticks([])
left, right = axes[1].set_xlim()
axes[1].hlines(y=50000,xmin=left, xmax=right, label= "200 Normseiten", color="red")
axes[1].set_ylim([0, 150000])
plt.text(right, 50000, "200 Normseiten", ha="right",va="center", color="red", rotation= 90)
plt.suptitle("Gattungs- und Medienabhängigkeit der Textlänge")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.boxplot([all_MLP_df.values.tolist(), R_df.values.tolist(), ], labels=["alle_MLP", "Roman"])
ax.set_ylabel("Textlänge in Wort-Token")
left, right = ax.set_xlim()
ax.hlines(y=50000,xmin=left, xmax=right, color="red")
ax.set_ylim([0, 150000])

plt.text(right, 50000, "200 Normseiten", ha="right",va="center", color="red", rotation= 90)
plt.title("Textsortenabhängigkeit der Textlänge")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.boxplot([ R_df.values.tolist(), MLP0_df.values.tolist(), otherE_df.values.tolist(), N_E_df.values.tolist()], labels=["Roman", "kein Label", "andere Label", "›Novelle‹ und \n ›Erzählung‹"])
ax.set_ylabel("Textlänge in Wort-Token")
left, right = ax.set_xlim()
ax.hlines(y=50000,xmin=left, xmax=right, color="red")
ax.set_ylim([0, 150000])

plt.text(right, 50000, "200 Normseiten", ha="right",va="center", color="red", rotation= 90)
plt.title("Textsortenabhängigkeit der Textlänge")

plt.tight_layout()
plt.show()


genre_grouped = df.groupby("Gattungslabel_ED_normalisiert")

df_red = df.drop(columns=["Medientyp_ED", "Jahr_ED"], axis=1)
grouped = df_red.groupby(["periods", "Gattungslabel_ED_normalisiert"]).mean()
grouped_df = pd.DataFrame(grouped)
#grouped_df = grouped_df.drop(columns=["Jahr_ED"], axis=1)
print(grouped_df)

fig, axes = plt.subplots(1,2, figsize=(12,5))
grouped_df.unstack().plot(kind='bar', stacked=False,
                                       title=str("Entwicklung der Textlänge über Gattungen"),

                          color=["green", "orange", "red", "blue", "yellow"],
                          ylabel=str("Länge in Wort-Token"), ax=axes[0])
left, right = axes[0].set_xlim()
axes[0].hlines(y=25000,xmin=left, xmax=right, color="red")
axes[0].set_ylim(0, 50000)
axes[0].set_xlabel("")

#grouped_df.unstack().plot(kind="line", stacked=False,
#            title=str("Entwicklung der Textlänge über Gattungen"),
#           xlabel="Zeitverlauf von 1770 bis 1950", ylabel=str("Länge in Wort-Token"))
#plt.tight_layout()
#plt.show()


media_df =  df[df.isin({"Medientyp_ED": ["Taschenbuch", "Familienblatt", "Anthologie", "Rundschau", "Buch"]}).any(axis=1)]
media_df_red = media_df.drop(columns=["Gattungslabel_ED_normalisiert"], axis=1)
grouped = media_df_red.groupby(["periods", "Medientyp_ED"]).median()
grouped_df = pd.DataFrame(grouped)
#grouped_df = grouped_df.drop(columns=["Jahr_ED"])

grouped_df.unstack().plot(kind='bar', stacked=False,
                                       title=str("Entwicklung der Textlänge über Medientypen"),
                                        color=["magenta", "cyan", "pink", "brown", "blue"],
                                      ax=axes[1])
left, right = axes[1].set_xlim()
axes[1].hlines(y=25000,xmin=left, xmax=right, color="red")
axes[1].set_ylim(0,50000)
axes[1].set_xlabel("")
plt.text(right, 25000, "100 Normseiten", ha="right",va="center", color="red", rotation= 90)
fig.supxlabel("Zeitverlauf")
#fig.text(0.5, 0.04, 'Zeitverlauf von 1790 bis 1950 (in 20-Jahres-Schritten)', ha='center')
plt.tight_layout()
plt.show()

print(grouped_df)
grouped_df.boxplot(by=["Medientyp_ED"], column=["token_count"])
plt.title("Boxplot, gruppiert nach Medientyp")
left, right = plt.xlim()
plt.hlines(y=25000,xmin=left, xmax=right, label= "100 Normseiten", color="red")
plt.text(right, 25000, "100 Normseiten", ha="center",va="center", rotation= 90, color="red")
plt.suptitle("")
plt.show()

grouped_df.unstack().plot(kind="line", stacked=False,
                      title=str("Length development for genres"),
                    xlabel="Zeitverlauf von 1770 bis 1950",
                          ylabel=str("Länge in Wort-Token"))
plt.tight_layout()
plt.show()

print(N_df.sort_values().iloc[-10:])

print(E_df.sort_values().iloc[-10:])

print(otherE_df.sort_values().iloc[-10:])

print(MLP0_df.sort_values().iloc[-10:])

print("Median novels:")
print(np.median(R_df.values))