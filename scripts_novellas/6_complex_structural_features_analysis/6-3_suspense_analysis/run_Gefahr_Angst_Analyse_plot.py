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
from sklearn.neighbors import NearestCentroid
from math import dist

genre_cat_name = "Gattungslabel_ED_normalisiert"

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


infile_path = os.path.join(local_temp_directory(system), "All_Chunks_Danger_FearCharacters_novellas.csv") # all chunks

matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)

df = matrix.data_matrix_df

df = df.rename(columns=columns_transl_dict)

print(df)

metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv" )
metadata_df = pd.read_csv(metadata_filepath, index_col=0)

print(metadata_df)

df[genre_cat_name] = df.apply(lambda x: metadata_df.loc[x["doc_id"], genre_cat_name], axis=1)
df[genre_cat_name] = df[genre_cat_name].fillna("unknown")
print(df)


y_variable = "Angstempfinden"
x_variables = ["max_value"]

if y_variable == "UnbekannteEindr":
    y_variable_legend = "Undefinierbare Eindrücke"
else: y_variable_legend = y_variable

infile_path = os.path.join(local_temp_directory(system), "MaxDangerFearCharacters_novellas.csv") # only chunk with max danger value for text
max_matrix = DocFeatureMatrix(data_matrix_filepath=infile_path, metadata_csv_filepath=metadata_filepath, metadata_df=None, mallet=False)
max_matrix = max_matrix.add_metadata(genre_cat_name)
df = max_matrix.data_matrix_df
df[genre_cat_name] = df[genre_cat_name].fillna("unknown")
print(df)

genres_list = ["N", "E", "R", "0E", "XE", "M"]
colors_list = ["red", "green", "blue", "cyan", "brown","yellow"]
zipped_dict = dict(zip(genres_list, colors_list[:len(colors_list)]))

mpatches_list = []
for genre, color in zipped_dict.items():
    patch = mpatches.Patch(color=color, label=genre.capitalize())
    mpatches_list.append(patch)

for x_variable in x_variables:
    fig, ax = plt.subplots()
    plt.scatter(df.loc[:, x_variable], df.loc[:, y_variable], color="grey")
    regr = LinearRegression()
    regr.fit(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])
    y_pred = regr.predict(df.loc[:, x_variable].array.reshape(-1, 1))
    plt.plot(df.loc[:, x_variable], y_pred, color="black", linewidth=3)

    if x_variable == "max_value":
        x_variable_legend = "Gefahrenlevel"
    else:
        x_variable_legend = x_variable

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.ylabel(y_variable_legend)
    plt.yticks(rotation=45)
    plt.xlabel(x_variable_legend)
    plt.title("Modell: Suspense als Korrelation von Gefahr und Angst")
    outfilename = "Modell_Suspense_Korrelation" + x_variable + y_variable + ".png"
    plt.savefig(os.path.join(local_temp_directory(system), "figures", outfilename))
    plt.show()

    clf = NearestCentroid()
    clf.fit(df[[x_variable, y_variable]].values, df[genre_cat_name])
    centroids = clf.centroids_
    print(centroids)
    class_labels = clf.classes_.tolist()

    new_colors_list = [zipped_dict[k] for k in class_labels if k in colors_list]
    ax.scatter(centroids[:, 0], centroids[:, 1], color=new_colors_list)

    df["dist_centroid"] = df.apply(lambda x: dist([x[x_variable], x[y_variable]],
                                                  [centroids[class_labels.index(x[genre_cat_name])][0],
                                                   centroids[class_labels.index(x[genre_cat_name])][1]]), axis=1)

    for i in range(len(class_labels)):
        class_label = class_labels[i]

        if class_label in genres_list:
            print(class_label)
            class_df = df[df[genre_cat_name] == class_label]

            class_av_dist = class_df["dist_centroid"].mean()
            print(class_av_dist)

            xy = (centroids[i])
            print(xy)

            color = zipped_dict[class_label]
            print(color)
            circle = plt.Circle(xy, 2 * class_av_dist, color=color, fill=False)
            ax.add_patch(circle)

            plt.annotate(text=str(class_labels[i]), xy=xy, color=color)



    fig, ax = plt.subplots()

    for genre, color in zipped_dict.items():
        df["doc_id"] = df.index
        df.index.name = None
        idx = df.groupby("doc_id")[x_variable].transform(max) == df[x_variable]
        print(idx)
        max_chunk_df = df[idx]

        genre_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=max_chunk_df, metadata_csv_filepath=metadata_filepath)
        genre_obj = genre_obj.reduce_to_categories(genre_cat_name, [genre])
        genre_obj = genre_obj.eliminate([genre_cat_name])
        genre_df = genre_obj.data_matrix_df
        print(genre_df)
        if color == "red":
            plt.scatter(genre_df.loc[:, x_variable], genre_df.loc[:, y_variable], marker= "x", color=color, label=genre)
        else:
            plt.scatter(genre_df.loc[:,x_variable], genre_df.loc[:, y_variable], marker = "o", color=color, label=genre, alpha=1)


    plt.xlabel(x_variable_legend)
    plt.ylabel(y_variable_legend)
    plt.title("Korrelation (für jeweils gefährlichsten Abschnitt)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles= mpatches_list)
    plt.show()

print("Finished!")