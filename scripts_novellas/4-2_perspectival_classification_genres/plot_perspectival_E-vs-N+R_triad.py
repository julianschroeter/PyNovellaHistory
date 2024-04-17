system = "my_xps" # "wcph113"

name_cat = "Nachname"
periods_cat = "Jahr_ED"
genre_cat = "Gattungslabel_ED_normalisiert"

import os
import pandas as pd
from preprocessing.presetting import local_temp_directory, global_corpus_representation_directory
from preprocessing.corpus import DocFeatureMatrix
from clustering.my_plots import plot_prototype_concepts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from math import sqrt, pow


infile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_E-R_n100.txt")
E_R_data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_E-R_n100.csv")
N_E_data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_N-E_n100.csv")
metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


with open(infile_output, "r") as f:
    lines = f.readlines()
optimum_x = str(lines[0]).split(": ")[1]
optimum_x = optimum_x.replace("\n", "")
optimum_x = float(optimum_x)
optimum_y = str(lines[1]).split(": ")[1]
print(optimum_x)
print(optimum_y)

obj = DocFeatureMatrix(data_matrix_filepath=E_R_data_matrix_filepath, metadata_csv_filepath=metadata_path)

obj = obj.add_metadata(periods_cat)
df = obj.data_matrix_df
df = df.rename(columns={"mean":"mean_R_E"})
df = df[df.isin({"Gattungslabel_ED_normalisiert":["E"]}).any(axis=1)]

N_E_df = pd.read_csv(N_E_data_matrix_filepath, index_col=0)
df["inv_N"] = df.apply(lambda x:  N_E_df.loc[x.name, "mean"], axis=1)
df["inv_R"] = df.apply(lambda x: 1 - x.mean_R_E, axis=1)

E_df = df.loc[:, ["inv_N", "inv_R", "Titel", "Nachname"]]
E_df = E_df.sort_values(by=["inv_R", "inv_N"])
E_df["centr_dist"] = E_df.apply(lambda x: sqrt(pow(x.inv_R,2) + pow(x.inv_N,2)), axis=1)
E_df["Kurztitel"] = E_df.apply(lambda x: str(x.Nachname) + ": " + str(x.Titel), axis=1)

labels = df[genre_cat].values.tolist()

subs_dict = {"N": "red", "E": "green"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

#subs_dict = {"red": 1, "green": 0}
#genre_bin = list(map(subs_dict.get, labels, labels))


#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]

fig, ax = plt.subplots(figsize=[8,8])

ax.scatter(E_df.inv_R, E_df.inv_N, c="green")
plt.annotate(E_df.iloc[0,5], (E_df.iloc[0,1], E_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

E_df = E_df.sort_values(by=["inv_R"], ascending=False)
plt.annotate(E_df.iloc[0,5], (E_df.iloc[0,1], E_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))


E_df = E_df.sort_values(by=["inv_N"])
#plt.annotate(N_df.iloc[0,5], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

E_df = E_df.sort_values(by=["inv_N"], ascending=False)
plt.annotate(E_df.iloc[0,5], (E_df.iloc[0,1], E_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

E_df = E_df.sort_values(by=["centr_dist"])
plt.annotate(E_df.iloc[0,5], (E_df.iloc[0,1], E_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

E_df = E_df.sort_values(by=["centr_dist"], ascending=False)
plt.annotate(E_df.iloc[0,5], (E_df.iloc[0,1], E_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate("E", (0,0), fontsize=30)
plt.annotate("R", (.95,0), fontsize=30)
plt.annotate("N", (0,.95), fontsize=30)

#plt.annotate("Mörike: Der Schatz", (df.loc["00367-00", periods_cat], df.loc["00367-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlim(0,1)
plt.ylim(0,1)
plt.axvline(x= 0.5 - ( optimum_x / 2), color = "blue")
plt.axvline(x=0.5 + ( optimum_x / 2), color= "blue")

infile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-E_n100.txt")
with open(infile_output, "r") as f:
    lines = f.readlines()
optimum_x = str(lines[0]).split(": ")[1]
optimum_x = optimum_x.replace("\n", "")
optimum_x = float(optimum_x)
optimum_y = str(lines[1]).split(": ")[1]
plt.axhline(y= 0.5 - ( optimum_x / 2), color="red")
plt.axhline(y=0.5 + ( optimum_x / 2), color="red")

plt.title("Distanz der Erzählungen zu Romanen und Novellen")
plt.xlabel("Erzählung --- Roman")
plt.ylabel("Erzählung ––– Novelle")

#mpatches_list = []
#for key, value in subs_dict.items():
#    patch = mpatches.Patch(color=value, label=key)
#    mpatches_list.append(patch)
#plt.legend(handles=mpatches_list)
plt.savefig(os.path.join("/home/julian/git/PyNovellaHistory/figures/E-N-R_perspektivisch_triadisch.svg"))
plt.show()

