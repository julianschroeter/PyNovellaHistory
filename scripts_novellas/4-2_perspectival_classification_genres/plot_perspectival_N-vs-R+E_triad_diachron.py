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


infile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-R-before1850_n100.txt")
N_R_data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_N-R-before1850_n100.csv")
N_E_data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_N-E_n100_before1850.csv")
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
N_df["Kurztitel"] = N_df.apply(lambda x: str(x.Nachname)[0] + ".: " + str(x.Titel)[:15]+".", axis=1)
N_df_bef = N_df.copy()
labels = df[genre_cat].values.tolist()

subs_dict = {"R": "blue", "N": "red"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

#subs_dict = {"red": 1, "green": 0}
#genre_bin = list(map(subs_dict.get, labels, labels))


#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]

fig, ax = plt.subplots(1,2, figsize=[12,6])

period_colors = ["orangered" if x == "1750-1850" else "red" for x in N_df.periods.tolist()]

ax[0].scatter(N_df.inv_N_R, N_df.inv_N_E, c=period_colors)
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

ax[0].annotate("N", (0,0), fontsize=30)
ax[0].annotate("R", (.95,0), fontsize=30)
ax[0].annotate("E", (0,.95), fontsize=30)

ax[0].annotate("N/E: VR-Enth: 1.02,\nVR-nicht-lin: 1.05", (0.1, 0.8), fontsize=10)
ax[0].annotate("N/R: VR-Enth: 1.04,\nVR-nicht-lin: 1.04", (0.9, 0.1), fontsize=10)

#ax[0].annotate("Mörike: Der Schatz", (df.loc["00367-00", periods_cat], df.loc["00367-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
ax[0].set_xlim(0,1)
ax[0].set_ylim(0,1)
ax[0].axvline(x= 0.5 - ( optimum_x / 2), color = "blue")
ax[0].axvline(x=0.5 + ( optimum_x / 2), color= "blue")

infile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-R-before1850_n100.txt")
with open(infile_output, "r") as f:
    lines = f.readlines()
optimum_x = str(lines[0]).split(": ")[1]
optimum_x = optimum_x.replace("\n", "")
optimum_x = float(optimum_x)
optimum_y = str(lines[1]).split(": ")[1]
ax[0].axhline(y= 0.5 - ( optimum_x / 2), color="green")
ax[0].axhline(y=0.5 + ( optimum_x / 2), color="green")

ax[0].set_title("1750–1850")
#ax[0].set_xlabel("Distanz zum Roman")
ax[0].set_ylabel("Distanz zur Erzählung")




infile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-R-after1850_n100.txt")
N_R_data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_N-R-after1850_n100.csv")
N_E_data_matrix_filepath = os.path.join(local_temp_directory(system), "av_pred_probabs_N-E_n100_after1850.csv")
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


period_colors = ["red" if x == "1750-1850" else "green" for x in N_df.periods.tolist()]

ax[1].scatter(N_df.inv_N_R, N_df.inv_N_E, c="red")
ax[1].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["inv_N_R"], ascending=False)
ax[1].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))


N_df = N_df.sort_values(by=["inv_N_E"])
ax[1].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["inv_N_E"], ascending=False)
ax[1].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["centr_dist"])
ax[1].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["centr_dist"], ascending=False)
ax[1].annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

ax[1].annotate("N", (0,0), fontsize=30)
ax[1].annotate("R", (.95,0), fontsize=30)
ax[1].annotate("E", (0,.95), fontsize=30)
ax[1].annotate("E/N: VR-Enth: 1.03, \nVR-nicht-lin: 0.98", (0.1,0.85), fontsize=10)
ax[1].annotate("N/R: VR-Enth: 1.03, \nVR-nicht-lin: 1,17", (0.7,0.1), fontsize=10)

#ax[1].annotate("Mörike: Der Schatz", (df.loc["00367-00", periods_cat], df.loc["00367-00", "mean"]), arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].set_xlim(0,1)
ax[1].set_ylim(0,1)
ax[1].axvline(x= 0.5 - ( optimum_x / 2), color = "blue")
ax[1].axvline(x=0.5 + ( optimum_x / 2), color= "blue")

infile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-R-after1850_n100.txt")
with open(infile_output, "r") as f:
    lines = f.readlines()
optimum_x = str(lines[0]).split(": ")[1]
optimum_x = optimum_x.replace("\n", "")
optimum_x = float(optimum_x)
optimum_y = str(lines[1]).split(": ")[1]
ax[1].axhline(y= 0.5 - ( optimum_x / 2), color="green")
ax[1].axhline(y=0.5 + ( optimum_x / 2), color="green")

ax[1].set_title("1850–1950")

ax[1].set_ylabel("Distanz zur Erzählung")

#mpatches_list = []
#for key, value in subs_dict.items():
#    patch = mpatches.Patch(color=value, label=key)
#    mpatches_list.append(patch)
#plt.legend(handles=mpatches_list)
fig.suptitle("Diachrone triadische perspektivische Modellierung")
fig.supxlabel("Distanz zum Roman")
fig.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system), "figures", "N_triadisch_perspektivisch_diachron.svg"))
plt.show()


