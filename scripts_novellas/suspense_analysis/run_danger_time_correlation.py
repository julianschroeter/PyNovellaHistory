system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/PyNovellaHistory')

import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, spearmanr
import numpy as np

genre_cat_name = "Gattungslabel_ED_normalisiert"
year_cat_name = "Jahr_ED"

scaler = StandardScaler()
scaler = MinMaxScaler()

columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"embedding_Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe", "Brustwarzen": "Erotik"}

dangers_list = ["Gewaltverbrechen", "Kampf", "Krieg", "Sturm", "Feuer", "Entführung"]
dangers_colors = ["cyan", "orange", "magenta", "blue", "pink", "purple"]
dangers_dict = dict(zip(dangers_list, dangers_colors[:len(dangers_list)]))

dangers_mpatches_list = []
for genre, color in dangers_dict.items():
    patch = mpatches.Patch(color=color, label=genre.capitalize())
    dangers_mpatches_list.append(patch)


infile_path = os.path.join(local_temp_directory(system),  "MaxDangerFearCharacters_novellas.csv") # all chunks

matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)

df = matrix.data_matrix_df

df = df.rename(columns=columns_transl_dict)

print(df)

metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv" )
metadata_df = pd.read_csv(metadata_filepath, index_col=0)

print(metadata_df)

df["doc_id"] = df.index
df.index.name = None
df.fillna(0, inplace=True)

df[genre_cat_name] = df.apply(lambda x: metadata_df.loc[x["doc_id"], genre_cat_name], axis=1)
df[year_cat_name] = df.apply(lambda x: metadata_df.loc[x["doc_id"], year_cat_name], axis=1)
df[genre_cat_name] = df[genre_cat_name].fillna("unknown")

df[year_cat_name] = df[year_cat_name].fillna(1828)

print(df)

genres_list = ["N", "E", "R", "0E", "XE", "M", "0P", "0PA", "0X_Essay", "0PB"]
genres_list = ["N", "E", "0E", "XE"]

print(df.isin({genre_cat_name: genres_list}).any(1))
df = df[df.isin({genre_cat_name: genres_list}).any(1)]
print(df[genre_cat_name].values)

colors_list = ["red", "green",  "pink", "pink","blue", "yellow", "cyan", "cyan", "cyan", "cyan"]
genres_dict = dict(zip(genres_list, colors_list[:len(genres_list)]))

genres_mpatches_list = []

for genre, color in genres_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    genres_mpatches_list.append(patch)

genre_colors_list = [genres_dict[x] for x in df[genre_cat_name].values.tolist()]
print(genre_colors_list)

y_variable = "max_value"
x_variables = [year_cat_name]

if y_variable == "max_value":
    y_variable_legend = "Maximum Gefahrenlevel im Text"
else: y_variable_legend = y_variable

for x_variable in x_variables:


    print("x and y variables are: ", x_variable, y_variable)
    print(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable].values)
    print("Spearman's rho: ", spearmanr(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])[0])
    print("Pearson's r: ", spearmanr(df.loc[:, x_variable], df.loc[:, y_variable])[0])
    fig, ax = plt.subplots()
    plt.scatter(df.loc[:, x_variable], df.loc[:, y_variable], color=genre_colors_list)
    regr = LinearRegression()
    regr.fit(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])
    y_pred = regr.predict(df.loc[:, x_variable].array.reshape(-1, 1))
    plt.plot(df.loc[:, x_variable], y_pred, color="black", linewidth=3)

    polymodel = np.poly1d(np.polyfit(df.loc[:, x_variable], df.loc[:, y_variable], 4))
    polyline = np.linspace(1800, 1920, 100)
    plt.plot(polyline, polymodel(polyline), color="green")

    if x_variable == year_cat_name:
        x_variable_legend = "Jahr des Erstdrucks"
    else:
        x_variable_legend = x_variable

    #plt.ylim(0, 1)
    #plt.xlim(0, 1)
    plt.ylabel(y_variable_legend)
    plt.yticks(rotation=45)
    plt.xlabel(x_variable_legend)
    plt.title("Korrelation von Zeit und Gefahrenlevel im Korpus")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=genres_mpatches_list)

    outfilename = "correlation_" + x_variable + y_variable + ".png"
    plt.savefig(os.path.join(local_temp_directory(system), "figures", outfilename))
    plt.show()
print("Finished!")