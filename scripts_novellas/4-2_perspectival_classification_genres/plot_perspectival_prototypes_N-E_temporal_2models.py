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


outfile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-E-before1850_n100.txt")
data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_N-E-before1850_n100.csv")
metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


with open(outfile_output, "r") as f:
    lines = f.readlines()
optimum_x = str(lines[0]).split(": ")[1]
optimum_x = optimum_x.replace("\n", "")
optimum_x = float(optimum_x)
optimum_y = str(lines[1]).split(": ")[1]
print(optimum_x)
print(optimum_y)

obj = DocFeatureMatrix(data_matrix_filepath=data_matrix_filepath, metadata_csv_filepath=metadata_path)

obj = obj.add_metadata(periods_cat)
df = obj.data_matrix_df

means = df["mean"].values.tolist()
years = df[periods_cat].values.tolist()
predict_probs_inv = [1 - value for value in means]
predict_probs = means
labels = df[genre_cat].values.tolist()

subs_dict = {"E": "green", "N": "red"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

#subs_dict = {"red": 1, "green": 0}
#genre_bin = list(map(subs_dict.get, labels, labels))


#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]

legend_dict = {"Novelle": "red", "Erzählung":"green"}

outfile_path = "/home/julian/git/PyNovellaHistory/figures/prototype_circle_N-E_before1850.svg"

fig, axes = plt.subplots(1,2, figsize=(12,5), width_ratios=[1,2])
axes[0].scatter(years, predict_probs, c=genre_c_labels)
axes[0].annotate("Arnim: Altdeutsche Landsleute", (df.loc["00292-00", periods_cat], df.loc["00292-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
#axes[0].annotate("Schnitzler: Leutnant Gustl", (df.loc["00494-00", periods_cat], df.loc["00494-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
axes[0].annotate("Neumann: Drei Fackeln", (df.loc["00094-00", periods_cat], df.loc["00094-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))

axes[0].axhline(y= 0.5 - ( optimum_x / 2))
axes[0].axhline(y=0.5 + ( optimum_x / 2))
axes[0].set_title("Vor 1850")
axes[0].set_ylim(0.2, 0.8)
axes[0].set_xlim(1800,1850)

mpatches_list = []
for key, value in legend_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)
#axes[0].legend(handles=mpatches_list)


outfile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-E-after1850_n100.txt")
data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_N-E-after1850_n100.csv")
metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


with open(outfile_output, "r") as f:
    lines = f.readlines()
optimum_x = str(lines[0]).split(": ")[1]
optimum_x = optimum_x.replace("\n", "")
optimum_x = float(optimum_x)
optimum_y = str(lines[1]).split(": ")[1]
print(optimum_x)
print(optimum_y)

obj = DocFeatureMatrix(data_matrix_filepath=data_matrix_filepath, metadata_csv_filepath=metadata_path)

obj = obj.add_metadata(periods_cat)
df = obj.data_matrix_df

means = df["mean"].values.tolist()
years = df[periods_cat].values.tolist()
predict_probs_inv = [1 - value for value in means]
predict_probs = means
labels = df[genre_cat].values.tolist()

subs_dict = {"E": "green", "N": "red"}
genre_c_labels = list(map(subs_dict.get, labels, labels))


legend_dict = {"Novelle": "red", "Erzählung":"green"}

outfile_path = "/home/julian/git/PyNovellaHistory/figures/prototype_circle_N-E_after1850.svg"

axes[1].scatter(years, predict_probs, c=genre_c_labels)
axes[1].annotate("Keller: Romeo/Julia", (df.loc["00306-00", periods_cat], df.loc["00306-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
axes[1].annotate("Schnitzler: Leutnant Gustl", (df.loc["00494-00", periods_cat], df.loc["00494-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
#axes[1].annotate("Neumann: Drei Fackeln", (df.loc["00094-00", periods_cat], df.loc["00094-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))

axes[1].axhline(y= 0.5 - ( optimum_x / 2))
axes[1].axhline(y=0.5 + ( optimum_x / 2))
axes[1].set_title("Nach 1850")
axes[1].set_ylim(0.2, 0.8 )
axes[1].set_xlim(1850,1950)

mpatches_list = []
for key, value in legend_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)
axes[1].legend(handles=mpatches_list)
fig.suptitle("Diachrone perspektivische Modellierung mit zeitlichem Verlauf")
fig.supxlabel("Zeitlicher Verlauf")
fig.supylabel("Vorhersagewahrscheinlichkeit")
fig.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system), "figures", "N-vs-E_diachron_perspectival_2models.svg"))
plt.show()



