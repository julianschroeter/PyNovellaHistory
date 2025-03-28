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

heftromane_infilepath = os.path.join(local_temp_directory(system), "MaxDanger_Heftromane_unscaled_with_metadata.csv")
novellas_infilepath = os.path.join(local_temp_directory(system),  "MaxDangerFearCharacters_novellas_unscaled.csv" )
novellas_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
heftromane_df = pd.read_csv(heftromane_infilepath, index_col=0)

heftromane_df= heftromane_df.assign(medium = "Heftroman")

heftromane_df = heftromane_df.drop(columns=["Figuren", "Figurenanzahl",
                                            "Netwerkdichte", "Anteil Figuren mit degree centrality == 1",
                                            "deg_centr", "weighted_deg_centr", "symp_dict", "author_norm", "id",
                                            "EndCharName_full", "symp_EndChar", "centr_EndChar",
                                            "weigh_centr_EndChar", "gender_EndChar","EndChar_series_protagonist",
                                            "GND", "series", "license", "publisher", "tokenCount",
                                            "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14" ])

novellas_df = pd.read_csv(novellas_infilepath, index_col = 0)
novellas_dtm_obj = DocFeatureMatrix(data_matrix_filepath=novellas_infilepath, metadata_csv_filepath=novellas_metadata_filepath)
novellas_dtm_obj = novellas_dtm_obj.add_metadata(["Titel", "Nachname","Jahr_ED","Gattungslabel_ED_normalisiert","Medium_ED", "Kanon_Status", "seriell"])
novellas_df = novellas_dtm_obj.data_matrix_df
novellas_df_full_media = standardize_meta_data_medium(df=novellas_df, medium_column_name="Medium_ED")

novellas_df = novellas_df_full_media.drop(columns=["Medium_ED", "medium"])

novellas_df = novellas_df.rename(columns={"Titel": "title", "Nachname":"author", "Jahr_ED":"date",
                                          "Gattungslabel_ED_normalisiert":"genre", "medium_type": "medium"})

whole_df = pd.concat([heftromane_df,novellas_df])

labels_list = ["R", "M", "E", "N", "0E", "XE", "0P", "0PB", "krimi", "abenteuer", "krieg"]
whole_df = whole_df[whole_df.isin({"genre": labels_list}).any(axis=1)]

other_cat_labels_list =  ["Taschenbuch", "Familienblatt", "Rundschau"]
other_cat_labels_list = ["Kleist", "Goethe", "Hoffmann" , "Eichendorff","Tieck", "Stifter", "Storm", "Keller", "Meyer", "Schnitzler", "Mann", "Musil"]
#whole_df = whole_df[whole_df.isin({"author": other_cat_labels_list}).any(1)]


replace_dict = {"genre": {"N": "MLP", "E": "MLP", "0E": "MLP", "XE": "MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen",
                          "krimi": "Spannungs-Heftroman", "abenteuer": "Spannungs-Heftroman", "krieg": "Spannungs-Heftroman"}}

replace_dict = {"genre": {"N": "Novelle", "E": "Erzählung", "0E": "sonst. MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen", "XE": "sonst. MLP"}}


whole_df = full_genre_labels(whole_df, replace_dict=replace_dict)

replace_dict = {"seriell": {"True": "Serie", "TRUE": "Serie", "vermutlich": "Serie",
                                                  "False": "nicht-seriell", "FALSE":"nicht-seriell"}}

whole_df = full_genre_labels(whole_df, replace_dict=replace_dict)

whole_df = whole_df.drop(whole_df['max_value'].idxmax())

serial_status_list = ["Serie", "nicht-seriell"]
#whole_df = whole_df[whole_df.isin({"seriell": serial_status_list}).any(axis=1)]

# coefficients for linear suspense model based on correlation in annotations: suspense = max_danger_level + 0.725 * Fear_level
whole_df["lin_susp_model"] = whole_df.apply(lambda x: x.max_value + (0.725 * x.Angstempfinden), axis=1)

scaler = MinMaxScaler()

scaled_values = scaler.fit_transform(whole_df[["Gewaltverbrechen", "Kampf", "Entführung", "Krieg", "Spuk","max_value", "Angstempfinden", "Sturm", "Feuer", "lin_susp_model"]])
whole_df[["Gewaltverbrechen", "Kampf", "Entführung", "Krieg","Spuk","max_value", "Angstempfinden", "Sturm", "Feuer", "lin_susp_model"]] = scaled_values
#whole_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

whole_df = years_to_periods(input_df=whole_df, category_name="date", start_year=1750,end_year=1900, epoch_length=50,
                            new_periods_column_name="periods")

whole_df = whole_df[whole_df["lin_susp_model"] != 0]
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



df = whole_df
year_cat_name = "date"
genre_cat_name = "genre"

df[year_cat_name] = df[year_cat_name].fillna(1828)

genres_list = ["N", "E", "R", "0E", "XE", "M", "0P", "0PA", "0X_Essay", "0PB"]

genres_list = ["MLP", "Märchen"] #, "Spannungs-Heftroman", "Roman"
genres_list = ["Novelle", "Erzählung", "sonst. MLP"]

df = df[df.isin({genre_cat_name: genres_list}).any(axis=1)]
df_serial = df[df.isin({"seriell": ["Serie"]}).any(axis=1)]
df_nonserial = df[df.isin({"seriell": ["nicht-seriell"]}).any(axis=1)]
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

serial_mpatches_list = []
#for serial, color in {"seriell publiziert":"black", "nicht-seriell publiziert":"grey"}.items():
for serial, color in {"serial": "black", "non-serial": "grey"}.items():
    patch = mpatches.Patch(color=color, label=serial)
    serial_mpatches_list.append(patch)

y_variable = "max_value" # "lin_susp_model"
x_variables = [year_cat_name]

if y_variable == "max_value":
    y_variable_legend = "Maximum Gefahrenlevel im Text"
    y_variable_legend = "Maximum Danger Level per Text"
elif y_variable == "lin_susp_model":
    y_variable_legend = "Baseline Modell: Gefahr-Angst-Spannung"
    y_variable_legend = "Baseline Modell: Danger-based Suspense"
else: y_variable_legend = y_variable


sns.lineplot(data=df, x="date", y="lin_susp_model", hue="seriell")
plt.title("Zeitliche Zu- und Abnahme von Spannung")
plt.ylabel("Baseline Modell: Gefahr-Angst-Spannung")
plt.xlabel("Jahr des Erstdrucks")
plt.show()


for x_variable in x_variables:


    print("x and y variables are: ", x_variable, y_variable)
    print(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable].values)
    print("Spearman's rho: ", spearmanr(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])[0])
    print("Pearson's r: ", pearsonr(df.loc[:, x_variable], df.loc[:, y_variable])[0])
    fig, ax = plt.subplots()

    plt.scatter(df_serial.loc[:, x_variable], df_serial.loc[:, y_variable], color="black")
    regr = LinearRegression()
    regr.fit(df_serial.loc[:, x_variable].array.reshape(-1, 1), df_serial.loc[:, y_variable])
    y_pred = regr.predict(df_serial.loc[:, x_variable].array.reshape(-1, 1))
    #plt.plot(df_serial.loc[:, x_variable], y_pred, color="black", linewidth=1, linestyle=":")

    # siegel-slope:
    x = df_serial.loc[:, x_variable]
    res = siegelslopes(df_serial.loc[:, y_variable], x)
    print(res)
    plt.plot(x, res[1] + res[0] * x, color="black", linewidth=3)

    plt.scatter(df_nonserial.loc[:, x_variable], df_nonserial.loc[:, y_variable], color="grey")
    regr = LinearRegression()
    regr.fit(df_nonserial.loc[:, x_variable].array.reshape(-1, 1), df_nonserial.loc[:, y_variable])
    y_pred = regr.predict(df_nonserial.loc[:, x_variable].array.reshape(-1, 1))
    #plt.plot(df_nonserial.loc[:, x_variable], y_pred, color="grey", linewidth=1, linestyle=":")

    # siegel-slope forgotten texts:
    x = df_nonserial.loc[:, x_variable]
    res = siegelslopes(df_nonserial.loc[:, y_variable], x)
    print(res)
    plt.plot(x, res[1] + res[0] * x, color="grey", linewidth=3)


    #poly_df = df[df[y_variable] != 3]
    #polymodel = np.poly1d(np.polyfit(poly_df.loc[:, x_variable], poly_df.loc[:, y_variable], 5))
    #polyline = np.linspace(1795, 1930, 135)
    #plt.plot(polyline, polymodel(polyline), color="green", linewidth=3)

    if x_variable == year_cat_name:
        x_variable_legend = "Jahr des Erstdrucks"
        x_variable_legend = "Year of Publication"
    else:
        x_variable_legend = x_variable

    #plt.ylim(0, 1)
    #plt.xlim(0, 1)
    plt.ylabel(y_variable_legend)
    plt.yticks(rotation=45)
    plt.xlabel(x_variable_legend)
    plt.title("Korrelation zwischen Zeit und Spannung für serielle/nicht-serielle Texte")
    plt.title("Correlation Between Time and Suspense for serial/non-serial Texts")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=serial_mpatches_list ) # authors_mpatches_list

    outfilename = "en_correlation_seriality" + x_variable + y_variable + ".svg"
    plt.savefig(os.path.join(local_temp_directory(system), "figures", outfilename))
    plt.show()



whole_df.boxplot(column="lin_susp_model", by="seriell")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

df.boxplot(column="lin_susp_model", by="medium")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="max_value", by="medium")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="max_value", by="genre")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


whole_df.boxplot(column="Angstempfinden", by="medium")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="Angstempfinden", by="genre")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

whole_df.boxplot(column="lin_susp_model", by="periods")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


whole_df.boxplot(column="max_value", by="periods")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


print("Finished!")