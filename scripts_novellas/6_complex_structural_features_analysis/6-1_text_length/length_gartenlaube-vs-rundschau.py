system = "my_xps"  # "wcph113" #
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

from preprocessing.presetting import global_corpus_representation_directory, language_model_path, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import years_to_periods, full_genre_labels, standardize_meta_data_medium
import matplotlib.pyplot as plt
import os
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
my_model_de = language_model_path(system)

infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
metadata_df_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_df_path, metadata_csv_filepath=metadata_df_path)
matrix_obj = matrix_obj.reduce_to(["token_count"])
matrix_obj = matrix_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED", "Medium_ED"])
matrix_obj.data_matrix_df = years_to_periods(matrix_obj.data_matrix_df, category_name="Jahr_ED", start_year=1850, end_year=1970, epoch_length=30, new_periods_column_name="periods")

cat_labels = ["N", "E", "0E", "XE"]
matrix_obj = matrix_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)


replace_dict = {"Medientyp_ED": {"Zeitschrift": "Journal", "Zeitung": "Journal",
                                 "Kalender": "Kalender", "Rundschau" : "Rundschau",
                                 "Zyklus" : "Anthologie", "Roman" : "Buch",
                                 "(unbekannt)" : "(unbekannt)",
                                    "Illustrierte": "Journal", "Sammlung": "Anthologie",
                                 "Nachlass": "Buch", "Jahrbuch":"Jahrbuch",
                                 "Monographie": "Buch", "Werke": "Buch"}}


#matrix_obj.data_matrix_df = full_genre_labels(matrix_obj.data_matrix_df, replace_dict=replace_dict)

matrix_obj.data_matrix_df = standardize_meta_data_medium(matrix_obj.data_matrix_df, "Medium_ED")


matrix_obj = matrix_obj.reduce_to_categories("periods", ["1850-1880"]) #, "1815-1822" , "1825-1845"
matrix_obj = matrix_obj.reduce_to_categories("medium", ["gartenlaube","dtrundsch"])

df = matrix_obj.data_matrix_df


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "kein Label",
                                    "R": "Roman", "M": "Märchen", "XE": "andere Label"}}
df = full_genre_labels(df, replace_dict=replace_dict)


#df = df[df["Medientyp_ED"] == "Anthologie"] # reduce to Taschenbücher
df = df[df["Jahr_ED"] >= 1790] # remove all rows where ...
df = df.drop(columns=["Medium_ED"], axis=1)

#df = df[df["Gattungslabel_ED_normalisiert"] != "Roman"] # remove all novels

# ANOVA with scipy.stats: calculte F-statistic and p-value
N_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'Novelle']
E_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'Erzählung']
otherE_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'andere Label']
MLP0_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'kein Label']
all_MLP_df = df[df.isin({"Gattungslabel_ED_normalisiert": ["Novelle", "Erzählung", "MLP0"]}).any(axis=1)]
N_E_df = df[df.isin({"Gattungslabel_ED_normalisiert": ["Novelle", "Erzählung"]}).any(axis=1)]
all_MLP_df = all_MLP_df.loc[:,"token_count"]
N_E_df = N_E_df.loc[:,"token_count"]

gartenlaube_N_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'Novelle'][df["medium"] == "gartenlaube"]
gartenlaube_E_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'Erzählung'][df["medium"] == "gartenlaube"]
gartenlaube_otherE_df = df["token_count"][df.isin({"Gattungslabel_ED_normalisiert": ["kein Label", "andere Label"]}).any(axis=1)][df["medium"] == "gartenlaube"]

daheim_N_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'Novelle'][df["medium"] == "dtrundsch"]
daheim_E_df = df['token_count'][df['Gattungslabel_ED_normalisiert'] == 'Erzählung'][df["medium"] == "dtrundsch"]
daheim_otherE_df = df["token_count"][df.isin({"Gattungslabel_ED_normalisiert": ["kein Label", "andere Label"]}).any(axis=1)][df["medium"] == "dtrundsch"]



F, p = stats.f_oneway(N_df, E_df)
print("F, p statistics of ANOVA test for Novellen versus Erzählungen:", F, p)
F, p = stats.f_oneway(N_df, otherE_df)
print("F, p statistics of ANOVA test for Novellen versus MLP:", F, p)

urania_df = df['token_count'][df['medium'] == 'urania']
aglaja_df = df['token_count'][df['medium'] == 'aglaja']

F, p = stats.f_oneway(urania_df, aglaja_df)
print("F, p statistics of ANOVA test for TB vs. Anth.:", F, p)

fig, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].boxplot([gartenlaube_N_df.values.tolist(), gartenlaube_E_df.values.tolist(), gartenlaube_otherE_df.values.tolist()], labels=["Novellen", "Erzählungen", "sonst_MLP"])
axes[0].set_title("Gartenlaube")
axes[0].set_ylabel("Textlänge in Wort-Token")
left, right = axes[0].set_xlim()
axes[0].hlines(y=25000,xmin=left, xmax=right, color="red")
axes[0].set_ylim([0, 50000])

axes[1].boxplot([daheim_N_df.values.tolist(), daheim_E_df.values.tolist(), daheim_otherE_df],
                labels=["Novellen", "Erzählungen", "sonst_MLP"])
axes[1].set_title("Deutsche Rundschau")
axes[1].set_yticks([])
left, right = axes[1].set_xlim()
axes[1].hlines(y=25000,xmin=left, xmax=right, label= "100 Normseiten", color="red")
axes[1].set_ylim([0, 50000])
plt.text(right, 25000, "100 Normseiten", ha="right",va="center", color="red", rotation= 90)
plt.suptitle("Gattungen in der Gartenlaube und in der Deutschen Rundschau 1850–1880")

plt.tight_layout()
plt.show()



genre_grouped = df.groupby("Gattungslabel_ED_normalisiert")

df_red = df.drop(columns=["Jahr_ED", "medium_type", "Gattungslabel_ED_normalisiert"], axis=1)
grouped = df_red.groupby(["periods","medium"]).mean()
grouped_df_all = pd.DataFrame(grouped)
#grouped_df = grouped_df.drop(columns=["Jahr_ED"], axis=1)
print(grouped_df_all)

fig, axes = plt.subplots(1,2, figsize=(12,5))

urania_df = df[df['medium'] == 'gartenlaube']
aglaja_df = df[df['medium'] == 'dtrundsch']
urania_df = urania_df.drop(columns=["medium", "medium_type", "Jahr_ED"], axis=1)
aglaja_df = aglaja_df.drop(columns=["medium", "medium_type", "Jahr_ED"], axis=1)

urania_grouped = urania_df.groupby(["periods", "Gattungslabel_ED_normalisiert"]).median()
grouped_df = pd.DataFrame(urania_grouped)
grouped_df.unstack().plot(kind='bar', stacked=False,
                                       title=str("Entwicklung der Textlänge: Prosa in der Gartenlaube"),

                          color=["green", "red", "orange", "blue", "yellow"],
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



aglaja_grouped = aglaja_df.groupby(["periods", "Gattungslabel_ED_normalisiert"]).median()
grouped_df = pd.DataFrame(aglaja_grouped)
#grouped_df = grouped_df.drop(columns=["Jahr_ED"])

grouped_df.unstack().plot(kind='bar', stacked=False,
                                       title=str("Entwicklung der Textlänge: Prosa in der Deutschen Rundschau"),
                                        color=["green", "red", "orange", "blue", "yellow"],
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


grouped_df_all.boxplot(by=["medium"], column=["token_count"])
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

