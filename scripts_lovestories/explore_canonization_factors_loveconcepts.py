system = "my_xps" #"wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/PyNovellaHistory')

import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr, siegelslopes
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np

year_cat_name = "Erscheinungsjahr"
genre_cat_name = "Gattungen"
media_cat = "Medium"

novellas_infilepath = outfile_path = os.path.join(global_corpus_representation_directory(system), "DocLoveconceptsMatrix_novellas.csv")

novellas_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

novellas_df = pd.read_csv(novellas_infilepath, index_col = 0)
novellas_dtm_obj = DocFeatureMatrix(data_matrix_filepath=novellas_infilepath, metadata_csv_filepath=novellas_metadata_filepath)
novellas_dtm_obj = novellas_dtm_obj.add_metadata(["Titel", "Nachname","Jahr_ED","Gattungslabel_ED_normalisiert","Medium_ED", "Kanon_Status", "seriell", "in_Deutscher_Novellenschatz"])
novellas_df = novellas_dtm_obj.data_matrix_df
novellas_df_full_media = standardize_meta_data_medium(df=novellas_df, medium_column_name="Medium_ED")

novellas_df = novellas_df_full_media.drop(columns=["Medium_ED", "medium"])

novellas_df = novellas_df.rename(columns={"Titel": "title", "Nachname":"author", "Jahr_ED":"Erscheinungsjahr",
                                          "Gattungslabel_ED_normalisiert":"Gattungen", "medium_type": "Medium",
                                          "seriell":"Serialität",
                                          "perfekt":"FinAmour", "unverheiratet":"Hochzeit",
                                          "einzigartig":"Individuelle_Liebe",
                                          "Bewunderung":"Generelle_Liebe",
                                          "ehelichen":"Ehe_Liebe", "leidenschaftlich":"AmourPassion"
                                          })

df = novellas_df


labels_list = ["R", "M", "E", "N", "0E", "XE", "0P", "0PB", "krimi", "abenteuer", "krieg"]
labels_list = ["R", "E", "N", "0E", "XE", "M"]
df = df[df.isin({genre_cat_name: labels_list}).any(axis=1)]

replace_dict = {genre_cat_name: {"N": "MLP", "E": "MLP", "0E": "MLP", "XE": "MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen",
                          "krimi": "Spannungs-Heftroman", "abenteuer": "Spannungs-Heftroman", "krieg": "Spannungs-Heftroman"}}

replace_dict = {genre_cat_name: {"N": "Novelle", "E": "Erzählung", "0E": "sonst. MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen", "XE": "sonst. MLP"}}


df = full_genre_labels(df, replace_dict=replace_dict)

replace_dict = {"Serialität": {"True": "Serie", "TRUE": "Serie", "vermutlich": "Serie",
                                                  "False": "nicht-seriell", "FALSE":"nicht-seriell"}}

df = full_genre_labels(df, replace_dict=replace_dict)


replace_dict = {"Kanon_Status": {0: "niedrig", 1: "niedrig", 2: "hoch",
                                                  3:"hoch"}}
df = full_genre_labels(df, replace_dict=replace_dict)


whole_df = df.copy()
serial_status_list = ["Serie", "nicht-seriell"]


df = whole_df

genres_list = ["N", "E", "R", "0E", "XE", "M", "0P", "0PA", "0X_Essay", "0PB"]

genres_list = ["MLP", "Märchen"] #, "Spannungs-Heftroman", "Roman"
genres_list = ["Novelle", "Erzählung", "sonst. MLP", "Roman", "Märchen"]


media_list = ["Taschenbuch", "Pantheon"]
media_list = ["Pantheon", "Journal", "Taschenbuch", "Familienblatt", "Rundschau", "Anthologie"]
media_list = ["Familienblatt", "Rundschau"]

media_df = df[df.isin({media_cat: media_list}).any(axis=1)]
df = df[df.isin({genre_cat_name: genres_list}).any(axis=1)]
df_serial = df[df.isin({"Serialität": ["Serie"]}).any(axis=1)]
df_nonserial = df[df.isin({"Serialität": ["nicht-seriell"]}).any(axis=1)]
df_canon = df[df.isin({"Kanon_Status": ["hoch"]}).any(axis=1)]

colors_list = ["cyan", "yellow", "pink", "blue", "green", "yellow", "cyan", "cyan", "cyan", "cyan"] # for genres
media_colors_list = ["darkgreen", "lightblue","pink", "grey",  "cyan", "red", "green", "yellow", "orange", "blue", "magenta", "black"]
colors_list = ["red", "green", "cyan", "blue", "orange" ]
genres_dict = dict(zip(genres_list, colors_list[:len(genres_list)]))

media_dict = dict(zip(media_list, media_colors_list[:len(media_list)]))
media_mpatches_list = []
for genre, color in media_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    media_mpatches_list.append(patch)

media_colors_list = [media_dict[x] for x in media_df[media_cat].values.tolist()]


genres_mpatches_list = []

for genre, color in genres_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    genres_mpatches_list.append(patch)


genre_colors_list = [genres_dict[x] for x in df[genre_cat_name].values.tolist()]


canon_colors_dict = {"hoch":"purple", "niedrig":"grey"}
canon_mpatches_list = []
for serial, color in canon_colors_dict.items():
    patch = mpatches.Patch(color=color, label=serial)
    canon_mpatches_list.append(patch)
canon_colors_list = [canon_colors_dict[x] for x in df["Kanon_Status"].values.tolist()]


y_variables = ["Ehe_Liebe", "AmourPassion","FinAmour", "Hochzeit", "Individuelle_Liebe"]
# y_variable = "Farben" #"Abstrakta" # "Hardseeds" # "fear_love" #"Liebe" # "max_value" #"Angstempfinden" #  "lin_susp_model"
x_variables = [year_cat_name ] # , "UnbekannteEindr"

loc_x_var, loc_y_var = "AmourPassion", "Hochzeit"
plt.scatter(media_df[loc_x_var], media_df[loc_y_var], c=media_colors_list)
# siegel-slope
x = media_df.loc[:, loc_x_var]
res = siegelslopes(media_df.loc[:, loc_y_var], x)
plt.plot(x, res[1] + res[0] * x, color="grey", linewidth=3)

plt.legend(handles=media_mpatches_list)  # authors_mpatches_list
plt.ylabel(loc_y_var)
plt.xlabel(loc_x_var)
#plt.xlim(0,0.8)
#plt.ylim(0,0.5)
plt.title("Korrelation von Gefahr und Angst in Medienformaten: " + loc_x_var + " " + loc_y_var)
plt.savefig("/home/julian/Documents/CLS_temp/figures/" + loc_x_var + "_" + "loc_y_var" + "_vor1850.svg")
plt.show()

plt.scatter(df[loc_x_var], df[loc_y_var], c=canon_colors_list)
# siegel-slope
x = df.loc[:, loc_x_var]
res = siegelslopes(df.loc[:, loc_y_var], x)
plt.plot(x, res[1] + res[0] * x, color="grey", linewidth=3)

plt.legend(handles=canon_mpatches_list)  # authors_mpatches_list
plt.ylabel(loc_y_var)
plt.xlabel(loc_x_var)
#plt.xlim(0,0.8)
#plt.ylim(0,0.5)
plt.title("Korrelation von Gefahr und Angst in Medienformaten_Kanon: " + loc_x_var + " " + loc_y_var)
plt.savefig("/home/julian/Documents/CLS_temp/figures/" + loc_x_var + "_" + "loc_y_var" + "Kanon_vor1850.svg")
plt.show()


for y_variable in y_variables:
    if y_variable == "max_value":
        y_variable_legend = "Gefahrenlevel im Text"
    elif y_variable == "lin_susp_model":
        y_variable_legend = "Baseline Modell: Gefahr-Angst-Spannung"
    else: y_variable_legend = y_variable

    sns.lineplot(data=df, x=year_cat_name, y=y_variable, hue="Kanon_Status",
                 palette={"niedrig":"grey", "hoch": "purple"})
    plt.title("Zeitliche Zu- und Abnahme: " + y_variable)
    plt.ylabel(y_variable)
    plt.xlabel("Jahr des Erstdrucks")
    plt.tight_layout()
    plt.savefig(os.path.join(local_temp_directory(system), "figures", y_variable + "_lineplot_canon_timeline.svg"))
    plt.show()

    sns.lineplot(data=df, x=year_cat_name, y=y_variable, hue="in_Deutscher_Novellenschatz",
                 palette={True:"orange", False: "grey"})
    plt.title("Zeitliche Zu- und Abnahme: " + y_variable)
    plt.ylabel(y_variable)
    plt.xlabel("Jahr des Erstdrucks")
    plt.tight_layout()
    plt.savefig(os.path.join(local_temp_directory(system), "figures", y_variable + "_lineplot_Novellenschatz_timeline.svg"))
    plt.show()


    for x_variable in x_variables:

        print("x and y variables are: ", x_variable, y_variable)
        res = spearmanr(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])
        print("Spearman's rho: ", res, res.pvalue)
        print("Pearson's r: ", pearsonr(df.loc[:, x_variable], df.loc[:, y_variable]))

        print("für serielle Texte:")
        res = spearmanr(df_serial.loc[:, x_variable].array.reshape(-1, 1), df_serial.loc[:, y_variable], alternative="greater")
        print("Spearman's rho: ", res, res.pvalue)

        print("für nicht-serielle Texte:")
        res = spearmanr(df_nonserial.loc[:, x_variable].array.reshape(-1, 1), df_nonserial.loc[:, y_variable],
                        alternative="less")
        print("Spearman's rho: ", res, res.pvalue)

        fig, ax = plt.subplots()

        plt.scatter(df.loc[:, x_variable], df.loc[:, y_variable], color="grey", alpha=0.5)
        regr = LinearRegression()
        regr.fit(df_serial.loc[:, x_variable].array.reshape(-1, 1), df_serial.loc[:, y_variable])
        y_pred = regr.predict(df_serial.loc[:, x_variable].array.reshape(-1, 1))
        #plt.plot(df_serial.loc[:, x_variable], y_pred, color="black", linewidth=1, linestyle=":")

        # siegel-slope
        x = df.loc[:, x_variable]
        res = siegelslopes(df.loc[:, y_variable], x)
        plt.plot(x, res[1] + res[0] * x, color="grey", linewidth=3)

        plt.scatter(df_canon.loc[:, x_variable], df_canon.loc[:, y_variable], color="purple")
        regr = LinearRegression()
        regr.fit(df_nonserial.loc[:, x_variable].array.reshape(-1, 1), df_nonserial.loc[:, y_variable])
        y_pred = regr.predict(df_nonserial.loc[:, x_variable].array.reshape(-1, 1))
       # plt.plot(df_nonserial.loc[:, x_variable], y_pred, color="grey", linewidth=1, linestyle=":")

        # siegel-slope canonical texts:
        x = df_canon.loc[:, x_variable]
        res = siegelslopes(df_canon.loc[:, y_variable], x)
        print(res)
        plt.plot(x, res[1] + res[0] * x, color="purple", linewidth=3)



        if x_variable == year_cat_name:
            x_variable_legend = "Jahr des Erstdrucks"
        else:
            x_variable_legend = x_variable

        #plt.ylim(0, 1)
        #plt.xlim(0, 1)
        plt.ylabel(y_variable_legend)
        plt.yticks(rotation=45)
        plt.xlabel(x_variable_legend)
        plt.title("Korrelation – Kanonisierung: Zeit + " + y_variable)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=canon_mpatches_list) # authors_mpatches_list

        outfilename = "correlation_seriality" + x_variable + y_variable + ".svg"
        plt.savefig(os.path.join(local_temp_directory(system), "figures", outfilename))
        plt.show()


        whole_df.boxplot(column=y_variable, by="in_Deutscher_Novellenschatz")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        whole_df.boxplot(column=y_variable, by="Kanon_Status")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


        media_df.boxplot(column=y_variable, by="Medium")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        whole_df.boxplot(column=y_variable, by=media_cat)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()




boxplot_colors = dict(boxes='black', whiskers='black', medians='gray', caps='black')



print("Finished!")