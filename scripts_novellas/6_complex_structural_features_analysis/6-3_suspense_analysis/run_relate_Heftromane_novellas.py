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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import numpy as np

heftromane_infilepath = os.path.join(local_temp_directory(system), "MaxDanger_Heftromane_unscaled_with_metadata.csv")
novellas_infilepath = os.path.join(local_temp_directory(system),  "AllChunksDangerFearCharacters_novellas_episodes_scaled.csv")
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
df = novellas_df
#df = df[df["doc_chunk_id"].map(len) == 8]
novellas_dtm_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=df, metadata_csv_filepath=novellas_metadata_filepath)
novellas_dtm_obj = novellas_dtm_obj.add_metadata(["Titel", "Nachname","Jahr_ED","Gattungslabel_ED_normalisiert","Medium_ED", "in_Pantheon"])
novellas_df = novellas_dtm_obj.data_matrix_df
novellas_df = standardize_meta_data_medium(df=novellas_df, medium_column_name="Medium_ED")


novellas_df = novellas_df.rename(columns={"Titel": "title", "Nachname":"author", "Jahr_ED":"date",
                                          "Gattungslabel_ED_normalisiert":"genre", "medium_type": "Medium"})

novellas_df["genre"] = novellas_df.apply(lambda x: "Pantheon" if x["in_Pantheon"] == True else x["genre"], axis=1)
novellas_df = novellas_df.drop(columns=["Medium_ED", "medium", "in_Pantheon"])


whole_df = pd.concat([heftromane_df,novellas_df])


labels_list = ["R", "M", "E", "N", "0E", "XE", "0P", "0PB", "krimi", "abenteuer", "krieg", "Pantheon"]
whole_df = whole_df[whole_df.isin({"genre": labels_list}).any(axis=1)]

other_cat_labels_list =  ["Taschenbuch", "Familienblatt", "Rundschau"]
other_cat_labels_list = ["Kleist", "Goethe", "Hoffmann" , "Eichendorff","Tieck", "Stifter", "Storm", "Keller", "Meyer", "Schnitzler", "Mann", "Musil"]
#whole_df = whole_df[whole_df.isin({"author": other_cat_labels_list}).any(axis=1)]

replace_dict = {"genre": {"N": "Novelle", "E": "Erzählung", "0E": "sonst. MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen", "XE": "sonst. MLP"}}

replace_dict = {"genre": {"N": "MLP", "E": "MLP", "0E": "MLP", "XE": "MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen",
                          "krimi": "Heftroman", "abenteuer": "Heftroman", "krieg": "Heftroman"}}


whole_df = full_genre_labels(whole_df, replace_dict=replace_dict)

whole_df = whole_df.drop(whole_df['max_value'].idxmax())

whole_df["lin_susp_model"] = whole_df.apply(lambda x: x.max_value + (0.725 * x.Angstempfinden), axis=1)


scaler = StandardScaler()
scaler = MinMaxScaler()

scaled_values = scaler.fit_transform(whole_df[["Gewaltverbrechen", "Kampf", "Entführung", "Krieg","max_value", "Angstempfinden", "Sturm", "Feuer", "lin_susp_model"]])
whole_df[["Gewaltverbrechen", "Kampf", "Entführung", "Krieg","max_value", "Angstempfinden", "Sturm", "Feuer", "lin_susp_model"]] = scaled_values
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
year_cat_name = "date"
genre_cat_name = "genre"

df[year_cat_name] = df[year_cat_name].fillna(1828)


print(df)

genres_list = ["N", "E", "R", "0E", "XE", "M", "0P", "0PA", "0X_Essay", "0PB"]
genres_list = ["Novelle", "Erzählung", "sonst. MLP"]
genres_list = ["MLP", "Märchen", "Heftroman", "Roman", "Pantheon"] #,


df = df[df.isin({genre_cat_name: genres_list}).any(axis=1)]

colors_list = ["red", "green", "cyan"]
colors_list = ["cyan", "yellow", "pink", "blue", "green", "yellow", "cyan", "cyan", "cyan", "cyan"] # for genres
colors_list = ["cyan", "orange", "black", "blue", "magenta", "blue", "pink", "grey", "magenta", "black", "darkgreen", "lightblue"]
genres_dict = dict(zip(genres_list, colors_list[:len(genres_list)]))

#authors_dict = dict(zip(other_cat_labels_list, colors_list[:len(other_cat_labels_list)]))

genres_mpatches_list = []

for genre, color in genres_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    genres_mpatches_list.append(patch)


genre_colors_list = [genres_dict[x] for x in df[genre_cat_name].values.tolist()]

authors_mpatches_list = []

#for genre, color in authors_dict.items():
 #   patch = mpatches.Patch(color=color, label=genre)
  #  authors_mpatches_list.append(patch)

#authors_colors_list = [authors_dict[x] for x in df["author"].values.tolist()]

y_variable = "max_value" #  "lin_susp_model"
x_variables = [year_cat_name]

if y_variable == "max_value":
    y_variable_legend = "Maximum Gefahrenlevel im Text"
elif y_variable == "lin_susp_model":
    y_variable_legend = "Baseline Modell: Gefahr-Angst-Spannung"
else: y_variable_legend = y_variable

for x_variable in x_variables:


    print("x and y variables are: ", x_variable, y_variable)
    print(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable].values)
    print("Spearman's rho: ", spearmanr(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])[0])
    print("Pearson's r: ", pearsonr(df.loc[:, x_variable], df.loc[:, y_variable])[0])
    fig, ax = plt.subplots()
    plt.scatter(df.loc[:, x_variable], df.loc[:, y_variable], color=genre_colors_list) # genre_colors_list
    #regr = LinearRegression()
    #regr.fit(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])
    #y_pred = regr.predict(df.loc[:, x_variable].array.reshape(-1, 1))
    #plt.plot(df.loc[:, x_variable], y_pred, color="black", linewidth=3)

    poly_df = df[df[y_variable] != 0]
    polymodel = np.poly1d(np.polyfit(poly_df.loc[:, x_variable], poly_df.loc[:, y_variable], 5))
    polyline = np.linspace(1795, 1930, 135)
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
    plt.title("Korrelation zwischen Zeit und Spannung")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=genres_mpatches_list) #  authors_mpatches_list

    outfilename = "correlation_" + x_variable + y_variable + ".svg"
    plt.savefig(os.path.join(local_temp_directory(system), "figures", outfilename))
    plt.show()

import seaborn as sns
sns.lineplot(data=df, x= x_variable, y=y_variable, hue="genre")
plt.show()

whole_df.boxplot(column="lin_susp_model", by="author")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

df.boxplot(column="lin_susp_model", by="genre")
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


print("Finished!")