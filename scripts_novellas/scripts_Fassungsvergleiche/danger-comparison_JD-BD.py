system = "my_xps"

from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory, global_corpus_raw_dtm_directory, global_corpus_directory
from preprocessing.corpus import DocFeatureMatrix
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind

corpus_path = global_corpus_directory(system)
metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

metadata_df = pd.read_csv(metadata_filepath, index_col=0)
print(metadata_df)
df = metadata_df.copy()
df = df.drop(labels="00311-00")
BD_df = df[df["spätere_Fassung_von"].notna()]

# drop works by author "Poe"
BD_df = BD_df[BD_df["Nachname"] != "Poe"]
BD_ids = BD_df.index.values.tolist()
ED_ids = BD_df["spätere_Fassung_von"].values.tolist()
# FV: Fassungsvergleiche
FV_ids = ED_ids + BD_ids

ED_df = df.loc[ED_ids]
print((ED_df))

input_doc_feautre_filepath = os.path.join(global_corpus_representation_directory(system), "MaxDangerMatrix_Fassungsvergleiche.csv")
dtm_obj = DocFeatureMatrix(data_matrix_filepath=input_doc_feautre_filepath, metadata_csv_filepath=metadata_filepath)

dtm_obj = dtm_obj.add_metadata(["Jahr_ED", "Nachname", "Titel", "Gattungslabel_ED_normalisiert"])

data = dtm_obj.data_matrix_df

data_ED = data.loc[ED_ids]
data_BD = data.loc[BD_ids]

data_ED = data_ED.reindex(labels=ED_ids)
data_BD = data_BD.reindex(labels=BD_ids)

print(data)

ED_values = data_ED.max_value.tolist()
BD_values = data_BD.max_value.tolist()

whiskerprops = dict(color="black", linewidth=1)
boxprops = dict(color="black", linewidth=1)
medianprops = dict(color="grey", linewidth=1)
fig, ax = plt.subplots()
ax.boxplot([ED_values, BD_values], medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops)
plt.xticks([1,2], ["Erstdrucke", "Buchdrucke"])
plt.xlabel("Medienformate")
plt.ylabel("Gefahrenlevel")
plt.title("Boxplots: Max. Gefahrenlevel – Fassungsvergleiche")
plt.savefig(os.path.join(local_temp_directory(system), "figures", "Abb_Boxplot_Fassungsvergleiche_Max_Gefahrenlevel_grey.svg"))
plt.show()

print(ttest_ind(ED_values, BD_values, alternative="greater"))
print(ttest_rel(ED_values, BD_values, alternative="greater"))

fig, ax=plt.subplots()
ax.scatter([0,1,2,3,4,5,6,7,8,9,10], ED_values, color="black", marker="x", alpha=0.8, label="Erstdrucke")
ax.scatter([0,1,2,3,4,5,6,7,8,9,10], BD_values, color="black",marker="o",  alpha=0.5, label="Buchdrucke")
plt.xticks([0,1,2,3,4,5,6,7,8,9,10], ["Stifter: Mantel/Bergmilch", "S.: Haidedorf", "S.: Heiliger Abend/Bergkr.",
                                      "Musil:Verz. Haus/Veronika", "Goethe: Melusine", "Schefer:Düvecke",
                                      "G.: Mann v. 50 Jahren", "S.: Mappe Urgroßv.", "Schiller: Geisterseher",
                                      "Marlitt: 12 Apostel", "S.: Condor"], rotation=90)
plt.legend()
plt.ylabel("Max. Gefahrenlevel")
plt.title("Wertepaare: Fassungsvergleich")
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures", "Abb_Wertepaare_Fassungsvergleiche_Max_Gefahrenlevel_grey.svg"))
plt.show()

data_ED = data_ED.loc[:,["Kampf", "Sturm", "Krieg", "Feuer", "Gewaltverbrechen"]]
data_ED["mean_values"] = data_ED.mean(axis=1)

data_BD = data_BD.loc[:,["Kampf", "Sturm", "Krieg", "Feuer", "Gewaltverbrechen"]]
data_BD["mean_values"] = data_BD.mean(axis=1)

ED_values = data_ED.mean_values.tolist()
BD_values = data_BD.mean_values.tolist()



fig, ax = plt.subplots()
ax.boxplot([ED_values, BD_values], medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops)
plt.xticks([1,2], ["Erstdrucke", "Buchdrucke"])
plt.xlabel("Medienformate")
plt.ylabel("Mittleres Gefahrenlevel")
plt.title("Boxplots: Mittleres Gefahrenlevel – Fassungsvergleiche")
plt.savefig(os.path.join(local_temp_directory(system), "figures", "Abb_Boxplot_Fassungsvergleiche_Mittleres_Gefahrenlevel_grey.svg"))
plt.show()

print(ttest_ind(ED_values, BD_values, alternative="greater"))
print(ttest_rel(ED_values, BD_values, alternative="greater"))

fig, ax=plt.subplots()
ax.scatter([0,1,2,3,4,5,6,7,8,9,10], ED_values, color="black", marker="x", alpha=0.8, label="Erstdrucke")
ax.scatter([0,1,2,3,4,5,6,7,8,9,10], BD_values, color="black", marker = "o", alpha=0.5, label="Buchdrucke")
plt.xticks([0,1,2,3,4,5,6,7,8,9,10], ["Stifter: Mantel/Bergmilch", "S.: Haidedorf", "S.: Heiliger Abend/Bergkr.",
                                      "Musil:Verz. Haus/Veronika", "Goethe: Melusine", "Schefer:Düvecke",
                                      "G.: Mann v. 50 Jahren", "S.: Mappe Urgroßv.", "Schiller: Geisterseher",
                                      "Marlitt: 12 Apostel", "S.: Condor"], rotation=90)
plt.legend()
plt.ylabel("Gefahrenlevel")
plt.title("Wertepaare: Fassungsvergleich")
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures", "Abb_Wertepaare_Fassungsvergleiche_Mittleres_Gefahrenlevel_grey.svg"))
plt.show()