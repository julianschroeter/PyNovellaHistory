system = "my_xps"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = "/home/julian/Documents/CLS_temp/tables/data_tables//hist_persp_modellierung_gattungspaare.csv"

df = pd.read_csv(filename, index_col = 0)
df = df.iloc[:3]
df

sns.lineplot(df.T, palette=["blue","green", "lightgreen"])
plt.title("Historische Perspektivische Modellierung: Gattungspaare")
plt.ylabel("Vorhersagegenauigkeit")
plt.xlabel("Zeit")
plt.savefig("/home/julian/git/PyNovellaHistory/figures/hist_persp_modellierung_gattungspaare.svg")
plt.show()

# make english version

sns.lineplot(df.T, palette=["orange","grey", "magenta"])
plt.title("Historical perspectival modeling for genre pairs")
plt.ylabel("Predictive accuracy")
plt.xlabel("Periods")
plt.savefig("/home/julian/git/PyNovellaHistory/figures/en_hist_persp_modellierung_gattungspaare.svg")
plt.show()

filename = "/home/julian/Documents/CLS_temp/tables/data_tables/auswertung_gattungspaare-poetiken.csv"

df = pd.read_csv(filename, index_col = 0, decimal=",")

df["Anteil_E_als_Modus"] = df.apply(lambda x: x.E_als_Modus / x.ges, axis=1)
df["Anteil_N_E_Synonym"] = df.apply(lambda x: x.N_syn_E / x.ges, axis=1)

df = df.iloc[:,[2,9,11,12]]

sns.lineplot(df)
plt.title("Abgrenzungen zum Begriff der Novelle in Poetiken")
plt.ylabel("Anteil der poetologischen Äußerungen")



df = df.iloc[:,:2]
df = df.rename(columns={"Anteil: N vs. E": "Novelle in Abgrenzung zur Erzählung", "Anteil: R vs N": "Novelle in Abgrenzung zum Roman"})

sns.lineplot(df, palette= ["magenta", "orange"])
plt.title("Poetologische Abgrenzungen vom Novellenbegriff")
plt.ylabel("Anteil der poetologischen Äußerungen")
plt.savefig("/home/julian/git/PyNovellaHistory/figures/Abgrenzung_Gattungspaare_Poetiken.svg")
plt.show()

df = df.rename(columns={"Novelle in Abgrenzung zur Erzählung": "Novelle as opposed to Erzählung", "Novelle in Abgrenzung zum Roman":"Novelle as opposed to Roman"})


# English version:

sns.lineplot(df, palette= ["magenta", "orange"])
plt.title("Emphasized genre delimitation in the poetics")
plt.ylabel("Proportion of delimitations in all poetological statements")
plt.savefig("/home/julian/git/PyNovellaHistory/figures/en_Abgrenzung_Gattungspaare_Poetiken.svg")
plt.show()
