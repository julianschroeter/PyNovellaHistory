system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')

import pandas as pd
from preprocessing.presetting import heftroman_base_directory, local_temp_directory
from preprocessing.corpus_alt import DocFeatureMatrix
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestCentroid
from math import dist


scaler = StandardScaler()
scaler = MinMaxScaler()
columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe"}

dangers_list = ["Gewaltverbrechen", "Kampf", "Krieg", "Sturm", "Feuer", "Entführung"]
dangers_colors = ["cyan", "orange", "magenta", "blue", "pink", "purple"]
dangers_dict = dict(zip(dangers_list, dangers_colors[:len(dangers_list)]))

dangers_mpatches_list = []
for genre, color in dangers_dict.items():
    patch = mpatches.Patch(color=color, label=genre.capitalize())
    dangers_mpatches_list.append(patch)


infile_path = os.path.join(local_temp_directory(system), "AllChunksDangerCharacters.csv")
metadata_filepath = os.path.join(heftroman_base_directory(system), "meta.tsv" )


matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)

df = matrix.data_matrix_df

df = df.rename(columns=columns_transl_dict)

metadata_filepath = os.path.join(heftroman_base_directory(system), "meta.tsv" )
metadata_df = pd.read_csv(metadata_filepath, sep="\t").set_index("id")
df["genre"] = df.apply(lambda x: metadata_df.loc[x["doc_id"], "genre"], axis=1)

y_variable = "Angstempfinden"
x_variables = ["max_value"]

if y_variable == "UnbekannteEindr":
    y_variable_legend = "Undefinierbare Eindrücke"
else: y_variable_legend = y_variable

infile_path = os.path.join(local_temp_directory(system), "MaxDangerCharacters.csv")
max_matrix = DocFeatureMatrix(data_matrix_filepath=infile_path, metadata_csv_filepath=metadata_filepath, metadata_df=None, mallet=False)
max_matrix = max_matrix.add_metadata("genre")
max_df = max_matrix.data_matrix_df

male_genres_list = ["krimi", "horror", "abenteuer", "scifi", "krieg", "fantasy", "western"]

max_df["gender"] = max_df["genre"].apply(lambda x: "Männergenres" if x in male_genres_list else "Frauengenres")

genres_list = ["krimi", "horror", "liebe", "abenteuer", "scifi", "krieg", "fantasy", "western"]
colors_list = ["green", "black", "red", "yellow", "cyan", "brown","purple", "orange"]
zipped_dict = dict(zip(genres_list, colors_list[:len(colors_list)]))

genres_list = ["krimi", "horror", "liebe", "abenteuer", "scifi", "krieg", "fantasy", "western"]
colors_list = ["blue", "red", "red", "blue", "blue", "blue","blue", "blue"]
gendered_genres_colors = dict(zip(genres_list, colors_list[:len(colors_list)]))

mpatches_list = []
for genre, color in zipped_dict.items():
    patch = mpatches.Patch(color=color, label=genre.capitalize())
    mpatches_list.append(patch)

gendered_colors_mpatches_list = []
genders = ["Frauengenres", "Männergenres"]
gender_colors = ["black", "grey"]
gendered_colors_dict = dict(zip(genders, gender_colors))
for genre, color in gendered_colors_dict.items():
    patch = mpatches.Patch(color=color, label=genre.capitalize())
    gendered_colors_mpatches_list.append(patch)

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
    clf.fit(df[[x_variable, y_variable]].values, df["genre"])
    centroids = clf.centroids_
    print(centroids)
    class_labels = clf.classes_.tolist()
    print(class_labels)

    new_colors_list = [zipped_dict[k] for k in class_labels if k in colors_list]
    ax.scatter(centroids[:, 0], centroids[:, 1], color=new_colors_list)

    df["dist_centroid"] = df.apply(lambda x: dist([x[x_variable], x[y_variable]],
                                                  [centroids[class_labels.index(x["genre"])][0],
                                                   centroids[class_labels.index(x["genre"])][1]]), axis=1)

    for i in range(len(class_labels)):
        class_label = class_labels[i]

        if class_label in genres_list:
            print(class_label)
            class_df = df[df["genre"] == class_label]

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
    #mpatches_list = []
    for gender, color in gendered_colors_dict.items():
        max_df["doc_id"] = max_df.index
        idx = max_df.groupby("doc_id")[x_variable].transform(max) == max_df[x_variable]
        print(idx)
        max_chunk_df = max_df[idx]

        #max_chunk_df.set_index("doc_chunk_id", inplace=True)

        genre_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=max_chunk_df, metadata_csv_filepath=metadata_filepath)


        genre_obj = genre_obj.reduce_to_categories("gender", [gender])
        genre_obj = genre_obj.eliminate(["genre"])
        genre_df = genre_obj.data_matrix_df
        print(genre_df)
        if color == "red":
            plt.scatter(genre_df.loc[:, x_variable], genre_df.loc[:, y_variable], marker= "x", color=color, label=gender)
        else:
            plt.scatter(genre_df.loc[:,x_variable], genre_df.loc[:, y_variable], marker = "o", color=color, label=gender, alpha=0.2)

        #regr = LinearRegression()
        #regr.fit(genre_df.loc[:,x_variable].array.reshape(-1,1), genre_df.loc[:, y_variable])
        #y_pred = regr.predict(genre_df.loc[:,x_variable].array.reshape(-1,1))
        #plt.plot(genre_df.loc[:,x_variable], y_pred, color = color, linewidth=3)

        #plt.show()
        #plt.scatter(krimis_df.loc[:,x_variable], krimis_df.loc[:, "Zweikampf"], color="green")
        #plt.show()

    plt.xlabel(x_variable_legend)
    plt.ylabel(y_variable_legend)
    plt.title("Korrelation (für jeweils gefährlichsten Abschnitt)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles= gendered_colors_mpatches_list)
    plt.show()