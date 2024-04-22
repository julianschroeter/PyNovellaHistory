system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/PyNovellaHistory')

import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr, siegelslopes
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns


novellas_infilepath = os.path.join(local_temp_directory(system),  "AllChunksDangerFearCharacters_novellas_episodes_scaled.csv")


novellas_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")



novellas_df = pd.read_csv(novellas_infilepath, index_col = 0)
novellas_dtm_obj = DocFeatureMatrix(data_matrix_filepath=novellas_infilepath, metadata_csv_filepath=novellas_metadata_filepath)
novellas_dtm_obj = novellas_dtm_obj.add_metadata(["Titel", "Nachname","Jahr_ED","Gattungslabel_ED_normalisiert","Medium_ED", "Kanon_Status", "seriell"])
novellas_df = novellas_dtm_obj.data_matrix_df
novellas_df_full_media = standardize_meta_data_medium(df=novellas_df, medium_column_name="Medium_ED")

novellas_df = novellas_df_full_media.drop(columns=["Medium_ED", "medium"])

novellas_df = novellas_df.rename(columns={"Titel": "title", "Nachname":"author", "Jahr_ED":"Erscheinungsjahr",
                                          "Gattungslabel_ED_normalisiert":"Gattungen", "medium_type": "Medium",
                                          "seriell":"Serialität"})

df = novellas_df
df = df[df["doc_chunk_id"].map(len) == 8]
print(df)

novellas_infilepath = os.path.join(local_temp_directory(system),  "MaxDangerFearCharacters_novellas_unscaled.csv" )
novellas_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
novellas_df = pd.read_csv(novellas_infilepath, index_col = 0)
novellas_dtm_obj = DocFeatureMatrix(data_matrix_filepath=novellas_infilepath, metadata_csv_filepath=novellas_metadata_filepath)
novellas_dtm_obj = novellas_dtm_obj.add_metadata(["Titel", "Nachname","Jahr_ED","Gattungslabel_ED_normalisiert","Medium_ED", "Kanon_Status"])
novellas_df = novellas_dtm_obj.data_matrix_df
novellas_df_full_media = standardize_meta_data_medium(df=novellas_df, medium_column_name="Medium_ED")
novellas_df = novellas_df_full_media.drop(columns=["Medium_ED", "medium"])
novellas_df = novellas_df.rename(columns={"Titel": "title", "Nachname":"author", "Jahr_ED":"date",
                                          "Gattungslabel_ED_normalisiert":"Gattungen", "medium_type": "Medium"})

#whole_df = pd.concat([heftromane_df,novellas_df])
whole_df = df.copy()

labels_list = ["R", "M", "E", "N", "0E", "XE", "0P", "0PB", "krimi", "abenteuer", "krieg"]
whole_df = whole_df[whole_df.isin({"Gattungen": labels_list}).any(axis=1)]

other_cat_labels_list =  ["Taschenbuch", "Familienblatt", "Rundschau"]
other_cat_labels_list = ["Kleist", "Goethe", "Hoffmann" , "Eichendorff","Tieck", "Stifter", "Storm", "Keller", "Meyer", "Schnitzler", "Mann", "Musil"]
#whole_df = whole_df[whole_df.isin({"author": other_cat_labels_list}).any(1)]


replace_dict = {"Gattungen": {"N": "MLP", "E": "MLP", "0E": "MLP", "XE": "MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen",
                          "krimi": "Spannungs-Heftroman", "abenteuer": "Spannungs-Heftroman", "krieg": "Spannungs-Heftroman"}}

replace_dict = {"Gattungen": {"N": "Novelle", "E": "Erzählung", "0E": "sonst. MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen", "XE": "sonst. MLP"}}




whole_df = full_genre_labels(whole_df, replace_dict=replace_dict)

replace_dict = {"Kanon_Status": {0: "vergessen", 1: "niedrig", 2: "mittel",
                                                  3: "hoch"}}
whole_df = full_genre_labels(whole_df, replace_dict=replace_dict)


#whole_df = whole_df.drop(whole_df['max_value'].idxmax())

whole_df["lin_susp_model"] = whole_df.apply(lambda x: x.max_value + (0.725 * x.Angstempfinden), axis=1)

scaler = MinMaxScaler()

scaled_values = scaler.fit_transform(whole_df[["Gewaltverbrechen", "Kampf", "Entführung", "Krieg", "Spuk","max_value", "Angstempfinden", "Sturm", "Feuer", "lin_susp_model"]])
whole_df[["Gewaltverbrechen", "Kampf", "Entführung", "Krieg","Spuk","max_value", "Angstempfinden", "Sturm", "Feuer", "lin_susp_model"]] = scaled_values
#whole_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)



df = whole_df.copy()

danger_df = df.drop(columns=["Erotik", "Liebe", "embedding_Angstempfinden","UnbekannteEindr", "Angstempfinden", "Liebe", "Erotik"])

danger_df["max_value"] = danger_df[["Gewaltverbrechen", "Kampf", "Entführung","Krieg","Sturm","Feuer"]].max(axis=1)
danger_df["max_danger_typ"] = danger_df[["Gewaltverbrechen", "Kampf", "Entführung","Krieg","Sturm","Feuer"]].idxmax(axis=1)

danger_df["embedding_Angstempfinden"] = df["embedding_Angstempfinden"]
danger_df["Angstempfinden"] = df["Angstempfinden"]
danger_df["UnbekannteEindr"] = df["UnbekannteEindr"]
danger_df["Liebe"] = df["Liebe"]
danger_df["Erotik"] = df["Erotik"]
danger_df["Sturm"] = df["Sturm"]
danger_df["Feuer"] = df["Feuer"]



print(whole_df)

df = whole_df
year_cat_name = "Erscheinungsjahr"
genre_cat_name = "Gattungen"

df[year_cat_name] = df[year_cat_name].fillna(1828)

print(df)

genres_list = ["N", "E", "R", "0E", "XE", "M", "0P", "0PA", "0X_Essay", "0PB"]

genres_list = ["MLP", "Märchen"] #, "Spannungs-Heftroman", "Roman"
genres_list = ["Novelle", "Erzählung", "sonst. MLP"]

df = df[df.isin({genre_cat_name: genres_list}).any(axis=1)]

can_status_list = ["hoch", "vergessen"]
df = df[df.isin({"Kanon_Status": can_status_list}).any(axis=1)]
#df = df[df["lin_susp_model"] != 0]
sns.lineplot(data=df, x=year_cat_name, y= "max_value", hue="Kanon_Status")
plt.title("Zeitliche Zu- und Abnahme von Spannung")
plt.ylabel("Baseline Modell: Gefahr-Angst-Spannung")
plt.xlabel("Jahr des Erstdrucks")
plt.show()

df_canon3 = df[df["Kanon_Status"]== "hoch"]
df_canon0 = df[df["Kanon_Status"]== "vergessen"]
colors_list = ["cyan", "yellow", "pink", "blue", "green", "yellow", "cyan", "cyan", "cyan", "cyan"] # for genres
colors_list = ["cyan", "red", "green", "yellow", "orange", "blue", "pink", "grey", "magenta", "black", "darkgreen", "lightblue"]
colors_list = ["red", "green", "cyan"]
genres_dict = dict(zip(genres_list, colors_list[:len(genres_list)]))

authors_dict = dict(zip(other_cat_labels_list, colors_list[:len(other_cat_labels_list)]))

genres_mpatches_list = []

for genre, color in genres_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    genres_mpatches_list.append(patch)


genre_colors_list = [genres_dict[x] for x in df[genre_cat_name].values.tolist()]

authors_mpatches_list = []

for genre, color in authors_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    authors_mpatches_list.append(patch)

#authors_colors_list = [authors_dict[x] for x in df["author"].values.tolist()]

canon_mpatches_list = []
for canon, color in {"vergessen":"grey", "hoch":"purple"}.items():
    patch = mpatches.Patch(color=color, label=canon)
    canon_mpatches_list.append(patch)

y_variable = "max_value" #"Liebe" #  # "lin_susp_model"
x_variables = [year_cat_name]

if y_variable == "max_value":
    y_variable_legend = "Maximum Gefahrenlevel im Text"
elif y_variable == "lin_susp_model":
    y_variable_legend = "Baseline Modell: Gefahr-Angst-Spannung"
else: y_variable_legend = y_variable



sns.lineplot(data=df, x=year_cat_name, y= y_variable, hue="Kanon_Status", palette={"hoch":"purple", "vergessen":"grey"})
plt.title("Zeitliche Zu- und Abnahme von " + y_variable)
plt.ylabel(y_variable)
plt.xlabel("Jahr des Erstdrucks")
plt.show()

for x_variable in x_variables:


    print("x and y variables are: ", x_variable, y_variable)
    print(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable].values)
    print("Spearman's rho: ", spearmanr(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])[0])
    print("Pearson's r: ", pearsonr(df.loc[:, x_variable], df.loc[:, y_variable])[0])
    fig, ax = plt.subplots()

    plt.scatter(df_canon0.loc[:, x_variable], df_canon0.loc[:, y_variable], color="grey") #  authors_colors_list
    regr = LinearRegression()
    regr.fit(df_canon0.loc[:, x_variable].array.reshape(-1, 1), df_canon0.loc[:, y_variable])
    y_pred = regr.predict(df_canon0.loc[:, x_variable].array.reshape(-1, 1))
    #plt.plot(df_canon0.loc[:, x_variable], y_pred, color="blue", linewidth=1, linestyle=":")

    # siegel-slope forgotten texts:
    x = df_canon0.loc[:, x_variable]
    res = siegelslopes(df_canon0.loc[:, y_variable], x)
    print(res)
    plt.plot(x, res[1] + res[0] * x, color="darkgrey", linewidth=3)

    plt.scatter(df_canon3.loc[:, x_variable], df_canon3.loc[:, y_variable], color="purple")  # authors_colors_list
    regr = LinearRegression()
    regr.fit(df_canon3.loc[:, x_variable].array.reshape(-1, 1), df_canon3.loc[:, y_variable])
    y_pred = regr.predict(df_canon3.loc[:, x_variable].array.reshape(-1, 1))
    #plt.plot(df_canon3.loc[:, x_variable], y_pred, color="orange", linewidth=1, linestyle=":")

    # siegel-slope Hochkanon:
    x = df_canon3.loc[:, x_variable]
    res = siegelslopes(df_canon3.loc[:, y_variable], x)
    print(res)
    plt.plot(x, res[1] + res[0] * x, color="purple", linewidth=3)


    #poly_df = df[df[y_variable] != 3]
    #polymodel = np.poly1d(np.polyfit(poly_df.loc[:, x_variable], poly_df.loc[:, y_variable], 5))
    #polyline = np.linspace(1795, 1930, 135)
    #plt.plot(polyline, polymodel(polyline), color="green", linewidth=3)

    if x_variable == year_cat_name:
        x_variable_legend = "Jahr des Erstdrucks"
    else:
        x_variable_legend = x_variable

    #plt.ylim(0, 1)
    #plt.xlim(0, 1)
    plt.ylabel(y_variable_legend)
    plt.yticks(rotation=45)
    plt.xlabel(x_variable_legend)
    plt.title("Korrelation zwischen Zeit und Spannung für Kanonisierungsstatus")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=canon_mpatches_list ) # authors_mpatches_list

    outfilename = "correlation_canon_" + x_variable + y_variable + ".svg"
    plt.savefig(os.path.join(local_temp_directory(system), "figures", outfilename))
    plt.show()



whole_df.boxplot(column="lin_susp_model", by="Kanon_Status")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

df.boxplot(column="lin_susp_model", by="Medium")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="max_value", by="Medium")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="max_value", by="Gattungen")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


whole_df.boxplot(column="Angstempfinden", by="Medium")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="Angstempfinden", by="Gattungen")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
print("Finished!")