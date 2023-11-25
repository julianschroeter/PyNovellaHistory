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
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt

media_name_cat = "Medium_ED"
medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"
pantheon_cat = "in_Pantheon"

scaler = StandardScaler()
scaler = MinMaxScaler()

columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"embedding_Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe", "Brustwarzen": "Erotik"}


dangers_list = ["Gewaltverbrechen", "Kampf", "Krieg", "Sturm", "Feuer", "Entführung"]
dangers_colors = ["cyan", "orange", "magenta", "blue", "pink", "purple"]
dangers_dict = dict(zip(dangers_list, dangers_colors[:len(dangers_list)]))


infile_path = os.path.join(local_temp_directory(system), "MaxDangerFearCharacters_novellas.csv") # "All_Chunks_Danger_FearCharacters_novellas.csv" # all chunks

matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)
df = matrix.data_matrix_df
df = df.rename(columns=columns_transl_dict)
df["doc_id"] = df.index
metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv" )
metadata_df = pd.read_csv(metadata_filepath, index_col=0)

matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=df, metadata_csv_filepath=metadata_filepath)
matrix_full = matrix.add_metadata(["Nachname", "Titel"])
df_full = matrix_full.data_matrix_df
matrix = matrix.reduce_to(["max_value"])
matrix = matrix.add_metadata([medium_cat, genre_cat, year_cat, pantheon_cat, "Kanon_Status", "in_Deutscher_Novellenschatz"])
df = matrix.data_matrix_df

df.rename(columns={"max_value": "dep_var"}, inplace=True) # "Netzwerkdichte"
dep_var = "dep_var"

df_pantheon = df["dep_var"][df[pantheon_cat] == True]
df_Novellenschatz = df["dep_var"][df["in_Deutscher_Novellenschatz"] == True]
genre_labels = ["N", "E", "0E", "XE"]
df = df[df.isin({genre_cat:genre_labels}).any(axis=1)]

replace_dict = {genre_cat: {"N": "N", "E": "E", "0E": "MLP",
                                    "R": "R", "M": "M", "XE": "MLP"}}
df = full_genre_labels(df, replace_dict=replace_dict)



df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1790, end_year=1950, epoch_length=20,
                      new_periods_column_name="periods")


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "kein Label",
                                    "R": "Roman", "M": "Märchen", "XE": "andere Label"}}
df = full_genre_labels(df, replace_dict=replace_dict)


df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)


# ANOVA with scipy.stats: calculte F-statistic and p-value
N_df = df["dep_var"][df['Gattungslabel_ED_normalisiert'] == 'Novelle']
E_df = df["dep_var"][df['Gattungslabel_ED_normalisiert'] == 'Erzählung']
otherE_df = df["dep_var"][df['Gattungslabel_ED_normalisiert'] == 'andere Label']
MLP0_df = df["dep_var"][df['Gattungslabel_ED_normalisiert'] == 'kein Label']
all_MLP_df = df[df.isin({"Gattungslabel_ED_normalisiert": ["Novelle", "Erzählung", "MLP0"]}).any(axis=1)]
N_E_df = df[df.isin({"Gattungslabel_ED_normalisiert": ["Novelle", "Erzählung"]}).any(axis=1)]
all_MLP_df = all_MLP_df.loc[:,"dep_var"]
N_E_df = N_E_df.loc[:,"dep_var"]
print(all_MLP_df)
R_df = df["dep_var"][df['Gattungslabel_ED_normalisiert'] == 'Roman']
M_df = df["dep_var"][df['Gattungslabel_ED_normalisiert'] == 'Märchen']
F, p = stats.f_oneway(N_df, E_df)
print("F, p statistics of ANOVA test for Novellen versus Erzählungen:", F, p)
F, p = stats.f_oneway(N_df, otherE_df)
print("F, p statistics of ANOVA test for Novellen versus MLP:", F, p)

TB_df = df["dep_var"][df['Medientyp_ED'] == 'Taschenbuch']
Ant_df = df["dep_var"][df['Medientyp_ED'] == 'Anthologie']
FB_df = df["dep_var"][df['Medientyp_ED'] == 'Familienblatt']
RS_df = df["dep_var"][df['Medientyp_ED'] == 'Rundschau']
Buch_df = df["dep_var"][df['Medientyp_ED'] == 'Buch']


F, p = stats.f_oneway(TB_df, Ant_df)
print("F, p statistics of ANOVA test for TB vs. Anth.:", F, p)
F, p = stats.f_oneway(FB_df, RS_df)
print("F, p statistics of ANOVA test for Familienblatt versus Rundschau:", F, p)

F, p = stats.f_oneway(df_pantheon, TB_df)
print("F, p statistics of ANOVA test for Pantheon versus TB:", F, p)

F, p = stats.f_oneway(df_pantheon, df_Novellenschatz)
print("F, p statistics of ANOVA test for Pantheon versus Novellenschatz:", F, p)

fig, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].boxplot([N_df.values.tolist(), E_df.values.tolist(), otherE_df.values.tolist(),
                  M_df.values.tolist()], labels=["›Novelle‹", "›Erzählung‹", "sonst_MLP",  "Märchen"])
axes[0].set_title("Gattungen")
axes[0].set_ylabel("Textlänge in Wort-Token")
left, right = axes[0].set_xlim()
#axes[0].hlines(y=50000,xmin=left, xmax=right, color="red")
axes[0].set_ylim([0, 1])

axes[1].boxplot([TB_df.values.tolist(), Ant_df.values.tolist(), FB_df.values.tolist(), RS_df.values.tolist(), Buch_df.values.tolist()],
                labels=["Taschenbuch", "Anthologie", "Familienblatt", "Rundschau", "Buch"])
axes[1].set_title("Medientypen")
axes[1].set_yticks([])
left, right = axes[1].set_xlim()
#axes[1].hlines(y=50000,xmin=left, xmax=right, label= "200 Normseiten", color="red")
axes[1].set_ylim([0, 1])
#plt.text(right, 50000, "200 Normseiten", ha="right",va="center", color="red", rotation= 90)
plt.suptitle("Gattungs- und Medienabhängigkeit der Textlänge")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.boxplot([all_MLP_df.values.tolist(), R_df.values.tolist(), ], labels=["alle_MLP", "Roman"])
ax.set_ylabel("Textlänge in Wort-Token")
left, right = ax.set_xlim()
#ax.hlines(y=50000,xmin=left, xmax=right, color="red")
ax.set_ylim([0, 1])
#plt.text(right, 50000, "200 Normseiten", ha="right",va="center", color="red", rotation= 90)
plt.title("Textsortenabhängigkeit der Textlänge")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.boxplot([df_pantheon.values.tolist(), TB_df.values.tolist(), df_Novellenschatz.values.tolist() ,RS_df.values.tolist()],
           labels=["in_Pantheon", "Taschenbuch", "in_Novellenschatz", "Rundschau"])
ax.set_ylabel("Textlänge in Wort-Token")
left, right = ax.set_xlim()
#ax.hlines(y=50000,xmin=left, xmax=right, color="red")
ax.set_ylim([0, 1])
#plt.text(right, 50000, "200 Normseiten", ha="right",va="center", color="red", rotation= 90)
plt.title("Textsortenabhängigkeit der Textlänge")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.boxplot([ R_df.values.tolist(), MLP0_df.values.tolist(), otherE_df.values.tolist(), N_E_df.values.tolist()], labels=["Roman", "kein Label", "andere Label", "›Novelle‹ und \n ›Erzählung‹"])
ax.set_ylabel("Textlänge in Wort-Token")
left, right = ax.set_xlim()
#ax.hlines(y=50000,xmin=left, xmax=right, color="red")
ax.set_ylim([0, 1])

#plt.text(right, 50000, "200 Normseiten", ha="right",va="center", color="red", rotation= 90)
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
axes[0].hlines(y=100000,xmin=left, xmax=right, color="red")
axes[0].set_ylim(0, 300000)
axes[0].set_xlabel("")


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
axes[1].hlines(y=100000,xmin=left, xmax=right, color="red")
axes[1].set_ylim(0,300000)
axes[1].set_xlabel("")
plt.text(right, 100000, "400 Normseiten", ha="right",va="center", color="red", rotation= 90)
fig.supxlabel("Zeitverlauf von 1790 bis 1950")
#fig.text(0.5, 0.04, 'Zeitverlauf von 1790 bis 1950 (in 20-Jahres-Schritten)', ha='center')
plt.tight_layout()
plt.show()

print(grouped_df)
grouped_df.boxplot(by=["Medientyp_ED"], column=["dep_var"])
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