system = "my_xps" # "wcph113"

name_cat = "Nachname"
periods_cat = "date"
genre_cat = "genre"

import os
import pandas as pd
from preprocessing.presetting import local_temp_directory, global_corpus_representation_directory
from preprocessing.corpus import DocFeatureMatrix
from clustering.my_plots import plot_prototype_concepts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


outfile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_Heftromane_n100.txt")
data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_Heftromane_n100.csv")
metadata_path = os.path.join(local_temp_directory(system), "Heftromane", "Metadatendatei_Heftromane.tsv")


with open(outfile_output, "r") as f:
    lines = f.readlines()
optimum_x = str(lines[0]).split(": ")[1]
optimum_x = optimum_x.replace("\n", "")
optimum_x = float(optimum_x)
optimum_y = str(lines[1]).split(": ")[1]
print(optimum_x)
print(optimum_y)

df = pd.read_csv(data_matrix_filepath, index_col=0)
print(df)

metadata_df = pd.read_csv(metadata_path, sep="\t", index_col=0)
df["date"] = df.apply(lambda x: metadata_df.loc[x.name, "date"], axis=1)

means = df["mean"].values.tolist()
years = df[periods_cat].values.tolist()
predict_probs_inv = [1 - value for value in means]
predict_probs = means
labels = df[genre_cat].values.tolist()

subs_dict = {"spannung": "blue", "liebe": "red"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

#subs_dict = {"red": 1, "green": 0}
#genre_bin = list(map(subs_dict.get, labels, labels))


#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]

plot_prototype_concepts(predict_probs_inv, genre_c_labels, threshold=optimum_x, annotation=None)

#romeo_prototyp = df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]
plot_prototype_concepts(predict_probs, genre_c_labels, threshold=optimum_x, annotation=None)

fig, ax = plt.subplots()
ax.scatter(years, predict_probs, c=genre_c_labels)
#plt.annotate("Keller: Romeo/Julia", (df.loc["00306-00", periods_cat], df.loc["00306-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
#plt.annotate("Eichend.: Ahnung/Gegenwart", (df.loc["00539-00", periods_cat], df.loc["00539-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
#plt.annotate("Mörike: Der Schatz", (df.loc["00367-00", periods_cat], df.loc["00367-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))

plt.axhline(y= 0.5 - ( optimum_x / 2))
plt.axhline(y=0.5 + ( optimum_x / 2))
plt.title("Predictive Probabilities from a historical perspective")

mpatches_list = []
for key, value in subs_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)
plt.legend(handles=mpatches_list)

plt.show()

