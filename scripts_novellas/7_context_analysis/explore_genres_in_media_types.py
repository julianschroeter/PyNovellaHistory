import pandas as pd

system = "my_xps"

from preprocessing.presetting import global_corpus_representation_directory
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
df = pd.read_csv(metadata_filepath, index_col=0)

df = standardize_meta_data_medium(df, "Medium_ED")

labels = ["N", "E", "0E", "XE"]
df = df[df.isin({"Gattungslabel_ED_normalisiert": labels}).any(axis=1)]


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "kein Label",
                                    "R": "Roman", "M": "Märchen", "XE": "andere Label"}}
df = full_genre_labels(df, replace_dict=replace_dict)

genre_colors_dict = {"Novelle": "red", "Erzählung": "green", "kein Label": "grey", "andere Label":"cyan"}

import matplotlib.patches as mpatches

mpatches_list = []
for key, value in genre_colors_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)


df_RS = df[df["medium"] == "dtrundsch"]
print(df_RS.Gattungslabel_ED_normalisiert)


df_urania = df[df["medium"] == "urania"]
df_urania.rename(columns={"Nachname":"Urania \n (1810–1840)"}, inplace=True)
grouped_urania = df_urania.groupby("Gattungslabel_ED_normalisiert").count().iloc[:,0]

df_agl = df[df["medium"] == "aglaja"]
df_agl.rename(columns={"Nachname":"Aglaja \n (1810–1840)"}, inplace=True)
grouped_agl = df_agl.groupby("Gattungslabel_ED_normalisiert").count().iloc[:,0]

df_conc = pd.concat([grouped_urania, grouped_agl], axis=1)
print(df_conc)
fig, axes = plt.subplots(3,2, figsize= (8,12))
grouped_urania.plot(kind="pie",
        colors=[genre_colors_dict[e] for e in grouped_urania.index.tolist()], autopct=lambda p: '{:.2f}%({:.0f})'.format(p,(p/100)*grouped_urania.sum()),
             ax=axes[1,0], textprops={'color':"w"}) #,

grouped_agl.plot(kind="pie",
        colors=[genre_colors_dict[e] for e in grouped_agl.index.tolist()], autopct=lambda p: '{:.2f}%({:.0f})'.format(p,(p/100)*grouped_agl.sum()),
             ax=axes[0,0], textprops={'color':"w"})


urania_genres = df_urania.Gattungslabel_ED_normalisiert.values.tolist()
print(Counter(urania_genres))





df_daheim = df[df["medium"] == "daheim"]
df_daheim.rename(columns={"Nachname":"FB Daheim \n (1864–1890)"}, inplace=True)
daheim_grouped = df_daheim.groupby("Gattungslabel_ED_normalisiert").count().iloc[:,0]
daheim_grouped.plot(kind="pie", autopct=lambda p: '{:.2f}%({:.0f})'.format(p,(p/100)*daheim_grouped.sum()),
        colors=[genre_colors_dict[e] for e in daheim_grouped.index.tolist()], ax=axes[1,1], textprops={'color':"w"})


df_gartlb = df[df["medium"] == "gartenlaube"]
df_gartlb.rename(columns={"Nachname":"FB Gartenlaube \n (1853–1890)"}, inplace=True)
gartlb_grouped = df_gartlb.groupby("Gattungslabel_ED_normalisiert").count().iloc[:,0]
gartlb_grouped.plot(kind="pie",
        colors=[genre_colors_dict[e] for e in gartlb_grouped.index.tolist()], ax=axes[0,1],
        autopct=lambda p: '{:.2f}%\n({:.0f})'.format(p,(p/100)*gartlb_grouped.sum()), textprops={'color':"w"})


df_pantheon = df[df["medium"] == "pantheon"]
df_pantheon.rename(columns={"Nachname":"Pantheon \n (1828–1830)"}, inplace=True)
pantheon_grouped = df_pantheon.groupby("Gattungslabel_ED_normalisiert").count().iloc[:,0]
pantheon_grouped.plot(kind="pie", autopct=lambda p: '{:.2f}%({:.0f})'.format(p,(p/100)*pantheon_grouped.sum()),
        colors=[genre_colors_dict[e] for e in pantheon_grouped.index.tolist()], ax=axes[2,0], textprops={'color':"w"})

df_dtRS = df[df["medium"] == "dtrundsch"]
df_dtRS.rename(columns={"Nachname":"Dt. Rundschau \n (1874–1890)"}, inplace=True)
dtRS_grouped = df_dtRS.groupby("Gattungslabel_ED_normalisiert").count().iloc[:,0]
dtRS_grouped.plot(kind="pie",
        colors=[genre_colors_dict[e] for e in dtRS_grouped.index.tolist()], ax=axes[2,1],
        autopct=lambda p: '{:.2f}%({:.0f})'.format(p,(p/100)*dtRS_grouped.sum()), textprops={'color':"w"})

fig.legend(handles=mpatches_list)
fig.suptitle("Verteilung der Gattungsbezeichnungen in einzelnen Organen    ")
fig.tight_layout()
plt.savefig("/home/julian/Documents/CLS_temp/figures/labels_in_medienformaten.svg")
plt.show()