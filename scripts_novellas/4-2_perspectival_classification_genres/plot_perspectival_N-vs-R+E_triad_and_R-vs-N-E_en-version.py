from setuptools.command.rotate import rotate

system = "my_xps" # "wcph113"

name_cat = "Nachname"
periods_cat = "Jahr_ED"
genre_cat = "Gattungslabel_ED_normalisiert"

import os
import pandas as pd
from preprocessing.presetting import local_temp_directory, global_corpus_representation_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import years_to_periods
from clustering.my_plots import plot_prototype_concepts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from math import sqrt, pow

# version for english plots:

infile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-R_n100.txt")
N_R_data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_N-R_n100.csv")
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

obj = DocFeatureMatrix(data_matrix_filepath=N_R_data_matrix_filepath, metadata_csv_filepath=metadata_path)

obj = obj.add_metadata(periods_cat)
df = obj.data_matrix_df
df = years_to_periods(df, category_name="Jahr_ED", start_year=1750, end_year=1950,epoch_length=100, new_periods_column_name="periods")


df = df.rename(columns={"mean":"mean_N_R"})
N_R_df_N = df[df.isin({"Gattungslabel_ED_normalisiert":["N"]}).any(axis=1)]

N_E_df = pd.read_csv(N_E_data_matrix_filepath, index_col=0)
N_R_df_N["inv_N_E"] = N_R_df_N.apply(lambda x: 1 - N_E_df.loc[x.name, "mean"], axis=1)
N_R_df_N["inv_N_R"] = N_R_df_N.apply(lambda x: 1 - x.mean_N_R, axis=1)

N_df = N_R_df_N.loc[:,["inv_N_E", "inv_N_R", "Titel", "Nachname", "periods", "Jahr_ED"]]
N_df = N_df.sort_values(by=["inv_N_R", "inv_N_E"])
N_df["centr_dist"] = N_df.apply(lambda x: sqrt(pow(x.inv_N_R,2) + pow(x.inv_N_E,2)), axis=1)
N_df["Kurztitel"] = N_df.apply(lambda x: str(x.Nachname)[0] + ".: " + str(x.Titel)[:15] + ".", axis=1)

labels = df[genre_cat].values.tolist()

subs_dict = {"R": "blue", "N": "red"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

#subs_dict = {"red": 1, "green": 0}
#genre_bin = list(map(subs_dict.get, labels, labels))


#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]

fig, ax = plt.subplots(1,2, figsize=(12,6))

period_colors = ["red" if x == "1750-1850" else "green" for x in N_df.periods.tolist()]

ax[0].scatter(N_df.inv_N_R, N_df.inv_N_E, c="red")
ax[0].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["inv_N_R"], ascending=False)
ax[0].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))


N_df = N_df.sort_values(by=["inv_N_E"])
ax[0].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["inv_N_E"], ascending=False)
ax[0].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["centr_dist"])
ax[0].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["centr_dist"], ascending=False)
ax[0].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

ax[0].annotate("N", (0.05,0.05), fontsize=30)
ax[0].annotate("R", (1,0.05), fontsize=30)
ax[0].annotate("E", (0.05,0.9), fontsize=30)

# insert the improvement rates (I did it manually, here, from the improvement rate calculations
#ax[0].annotate("N/E: VR-Enth: 1.01 \n VR-nicht-lin: 1.00", (0,0.85), fontsize=12) # mit Enthaltungsbereich für nicht-lineare Modelle (Familienähnlichkeit)
ax[0].text(0.01,0.55,"N/E: Improvment non-dec: 1.01",  fontsize=10, rotation=90)
ax[0].annotate("N/R: Impr. non-dec: 1.025", (0.55, 0.01), fontsize=10)
#ax[0].annotate("Mörike: Der Schatz", (df.loc["00367-00", periods_cat], df.loc["00367-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
ax[0].set_xlim(0,1)
ax[0].set_ylim(0,1)
ax[0].axvline(x= 0.5 - ( optimum_x / 2), color = "blue")
ax[0].axvline(x=0.5 + ( optimum_x / 2), color= "blue")

infile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-E_n100.txt")
with open(infile_output, "r") as f:
    lines = f.readlines()
optimum_x = str(lines[0]).split(": ")[1]
optimum_x = optimum_x.replace("\n", "")
optimum_x = float(optimum_x)
optimum_y = str(lines[1]).split(": ")[1]
ax[0].axhline(y= 0.5 - ( optimum_x / 2), color="green")
ax[0].axhline(y=0.5 + ( optimum_x / 2), color="green")

ax[0].set_title("Novelle and \n neighboring genres")
ax[0].set_xlabel("Novelle <---> Roman")
ax[0].set_ylabel("Novelle <---> Erzählung")

#mpatches_list = []
#for key, value in subs_dict.items():
#    patch = mpatches.Patch(color=value, label=key)
#    mpatches_list.append(patch)
#plt.legend(handles=mpatches_list)





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
E_df["Kurztitel"] = E_df.apply(lambda x: str(x.Nachname)[0] + ".: " + str(x.Titel)[:15] + ".", axis=1)

labels = df[genre_cat].values.tolist()

subs_dict = {"N": "red", "E": "green"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

#subs_dict = {"red": 1, "green": 0}
#genre_bin = list(map(subs_dict.get, labels, labels))


#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]


ax[1].scatter(E_df.inv_R, E_df.inv_N, c="green")

# ax[1].annotate("E/R: VR-Enth: 1.04 \n VR-nicht-lin: 1.16", (0.85, 0.1), fontsize=12) # mit Verbesserungsrate für nicht-lineare Modelle

# improvement rate non-linear SVM to linear SVM: 1.18 !!! -> Non-Linearity is stable for the comparisons with novels.
ax[1].text(0.01,0.55,"N/E: Improvment non-dec: 1.01",  fontsize=10, rotation=90)
ax[1].annotate("E/R: Impr. non-dec: 1.04", (0.6, 0.01), fontsize=10)

ax[1].annotate(E_df.iloc[0,5], (E_df.iloc[0,1], E_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

E_df = E_df.sort_values(by=["inv_R"], ascending=False)
ax[1].annotate(E_df.iloc[0,5], (E_df.iloc[0,1], E_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))


E_df = E_df.sort_values(by=["inv_N"])
#ax[1].annotate(N_df.iloc[0,5], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

E_df = E_df.sort_values(by=["inv_N"], ascending=False)
ax[1].annotate(E_df.iloc[0,5], (E_df.iloc[0,1], E_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

E_df = E_df.sort_values(by=["centr_dist"])
ax[1].annotate(E_df.iloc[0,5], (E_df.iloc[0,1], E_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

E_df = E_df.sort_values(by=["centr_dist"], ascending=False)
ax[1].annotate(E_df.iloc[0,5], (E_df.iloc[0,1], E_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

ax[1].annotate("E", (0.05,0.05), fontsize=30)
ax[1].annotate("R", (1,0.05), fontsize=30)
ax[1].annotate("N", (0.05,.9), fontsize=30)

#ax[1].annotate("Mörike: Der Schatz", (df.loc["00367-00", periods_cat], df.loc["00367-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].set_xlim(0,1)
ax[1].set_ylim(0,1)
ax[1].axvline(x= 0.5 - ( optimum_x / 2), color = "blue")
ax[1].axvline(x=0.5 + ( optimum_x / 2), color= "blue")

infile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-E_n100.txt")
with open(infile_output, "r") as f:
    lines = f.readlines()
optimum_x = str(lines[0]).split(": ")[1]
optimum_x = optimum_x.replace("\n", "")
optimum_x = float(optimum_x)
optimum_y = str(lines[1]).split(": ")[1]
ax[1].axhline(y= 0.5 - ( optimum_x / 2), color="red")
ax[1].axhline(y=0.5 + ( optimum_x / 2), color="red")

ax[1].set_title("Erzählung and \n neighboring genres")
ax[1].set_xlabel("Erzählung <---> Roman")
ax[1].set_ylabel("Erzählung <–––> Novelle")

#mpatches_list = []
#for key, value in subs_dict.items():
#    patch = mpatches.Patch(color=value, label=key)
#    mpatches_list.append(patch)
#plt.legend(handles=mpatches_list)
fig.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system), "figures", "en_N-E_perspektivisch_triadisch_2subplots_ohne_VR-nicht-lin.svg"))
plt.show()

