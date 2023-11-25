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
N_df["Kurztitel"] = N_df.apply(lambda x: str(x.Nachname) + ": " + str(x.Titel), axis=1)

labels = df[genre_cat].values.tolist()

subs_dict = {"R": "blue", "N": "red"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

#subs_dict = {"red": 1, "green": 0}
#genre_bin = list(map(subs_dict.get, labels, labels))


#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]

fig, ax = plt.subplots(figsize=[8,8])

period_colors = ["red" if x == "1750-1850" else "green" for x in N_df.periods.tolist()]

ax.scatter(N_df.inv_N_R, N_df.inv_N_E, c="red")
plt.annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["inv_N_R"], ascending=False)
plt.annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))


N_df = N_df.sort_values(by=["inv_N_E"])
plt.annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["inv_N_E"], ascending=False)
plt.annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["centr_dist"])
plt.annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

N_df = N_df.sort_values(by=["centr_dist"], ascending=False)
plt.annotate(N_df.iloc[0,7], (N_df.iloc[0,1], N_df.iloc[0,0]), arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate("N", (0,0), fontsize=30)
plt.annotate("R", (.95,0), fontsize=30)
plt.annotate("E", (0,.95), fontsize=30)

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
plt.axhline(y= 0.5 - ( optimum_x / 2), color="green")
plt.axhline(y=0.5 + ( optimum_x / 2), color="green")

plt.title("Distanz der Novellen zu Erzählung und Roman")
plt.xlabel("Distanz zum Roman")
plt.ylabel("Distanz zur Erzählung")

#mpatches_list = []
#for key, value in subs_dict.items():
#    patch = mpatches.Patch(color=value, label=key)
#    mpatches_list.append(patch)
#plt.legend(handles=mpatches_list)
plt.savefig(os.path.join("/home/julian/git/PyNovellaHistory/figures/N-E-R_perspektivisch_triadisch.svg"))
plt.show()

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
x = N_df.inv_N_R
y=  N_df.inv_N_E
z = N_df.Jahr_ED
ax.scatter3D(x,y,z, c=period_colors)

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)