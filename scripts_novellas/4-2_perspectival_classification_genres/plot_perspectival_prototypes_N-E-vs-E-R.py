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


outfile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-E_n100.txt")
data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_N-E_n100.csv")
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


fig, axes = plt.subplots(1,2, figsize=(12,5))

outfile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-R_n100.txt")
data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_N-R_n100.csv")
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

subs_dict = {"R": "blue", "N": "red"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

#subs_dict = {"red": 1, "green": 0}
#genre_bin = list(map(subs_dict.get, labels, labels))


#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]

legend_dict = {"Novelle": "red", "Roman":"blue"}

outfile_path = "/home/julian/git/PyNovellaHistory/figures/prototype_circle_N-R.svg"
plot_prototype_concepts(predict_probs_inv, genre_c_labels, threshold=optimum_x, annotation=None,
                        lang="de", legend_dict=legend_dict, filepath=outfile_path)

#romeo_prototyp = df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]
plot_prototype_concepts(predict_probs, genre_c_labels, threshold=optimum_x, annotation=None, lang="de", legend_dict=legend_dict)

axes[0].scatter(years, predict_probs, c=genre_c_labels)
axes[0].annotate("Der Einsiedler", (df.loc["00603-00", periods_cat], df.loc["00603-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
axes[0].annotate("Ahnung/Gegenwart", (df.loc["00539-00", periods_cat], df.loc["00539-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
#axes[0].annotate("Mörike: Der Schatz", (df.loc["00367-00", periods_cat], df.loc["00367-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
#axes[0].annotate("S.: Paralyse", (df.loc["00710-00", periods_cat], df.loc["00710-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
axes[0].annotate("Lieutenant Gustl", (df.loc["00494-00", periods_cat], df.loc["00494-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))

axes[0].axhline(y= 0.5 - ( optimum_x / 2))
axes[0].axhline(y=0.5 + ( optimum_x / 2))
axes[0].set_title("Novellen vs. Romane")
axes[0].set_ylim(0.05, 0.95 )
axes[0].set_xlim(1760,1940)
mpatches_list = []
for key, value in legend_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)
axes[0].legend(handles=mpatches_list)


outfile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_R-E_n100.txt")
data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_R-E_n100.csv")
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

subs_dict = {"R": "blue", "E": "green"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

#subs_dict = {"red": 1, "green": 0}
#genre_bin = list(map(subs_dict.get, labels, labels))


#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]

legend_dict = {"Erzählung": "green", "Roman":"blue"}

outfile_path = "/home/julian/git/PyNovellaHistory/figures/prototype_circle_E-R.svg"

axes[1].scatter(years, predict_probs, c=genre_c_labels)
#axes[1].annotate("Der Einsiedler ...", (df.loc["00603-00", periods_cat], df.loc["00603-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
axes[1].annotate("›Ahnung/Gegenwart‹", (df.loc["00539-00", periods_cat], df.loc["00539-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
#axes[1].annotate("Mörike: Der Schatz", (df.loc["00367-00", periods_cat], df.loc["00367-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
#axes[1].annotate("S.: Paralyse", (df.loc["00710-00", periods_cat], df.loc["00710-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
axes[1].annotate("Keller: ›Romeo/Julia‹", (df.loc["00306-00", periods_cat], df.loc["00306-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))

axes[1].axhline(y= 0.5 - ( optimum_x / 2))
axes[1].axhline(y=0.5 + ( optimum_x / 2))
axes[1].set_title("Romane vs. Erzählungen")
axes[1].set_ylim(0.05, 0.95 )
axes[1].set_xlim(1760,1940)

mpatches_list = []
for key, value in legend_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)
axes[1].legend(handles=mpatches_list)
fig.suptitle("Synchrones perspektivisches Modell mit zeitlichem Verlauf für Gattungspaare")
fig.supxlabel("Zeitlicher Verlauf")
fig.supylabel("Vorhersagewahrscheinlichkeit")
fig.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system), "figures", "N-R-vs-R-E-temporal_perspectival.svg"))
plt.show()

