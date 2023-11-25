import numpy as np

system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# from my own modules:
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods

# standard libraries
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import chi2_contingency

medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"


infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
metadata_df_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_df_path, metadata_csv_filepath=metadata_df_path)
matrix_obj = matrix_obj.reduce_to(["token_count"])
matrix_obj = matrix_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED", "Medientyp_ED"])

cat_labels = ["N", "E", "0E", "XE", "R", "M"]
matrix_obj = matrix_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)

df = matrix_obj.data_matrix_df

df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]
scaler = MinMaxScaler()
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

data = df.loc[:, ("token_count",genre_cat,medium_cat, "Jahr_ED")]


genre_selections = ["all_genres", "N-vs-E-vs-MLP", "N-vs-MLP", "nonR-vs-R", "N-vs-E"]
media_selections = ["all_media", "TB-Ant-RS-FB", "TB-Ant-RS-FB-Journal", "TB-Journal-Book", "Buch-vs-NichtBuch"]

tuples = []
for j in genre_selections:
    for k in media_selections:
        tuple = (j, k)
        tuples.append(tuple)

best_model_score = 0
all_model_params = []
for j,k in tuples:

    df = data.copy()


    if j == "all_genres":
        cat_labels = ["N", "E", "0E", "XE", "R", "M"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(axis=1)]
    elif j == "N-vs-E-vs-MLP":
        cat_labels = ["N", "E", "0E", "XE"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(axis=1)]
        replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "N", "E": "E", "0E": "MLP",
                                                        "XE": "MLP"}}
        df = full_genre_labels(df, replace_dict=replace_dict)
    elif j == "N-vs-MLP":
        cat_labels = ["N", "E", "0E", "XE"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(axis=1)]
        replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "N", "E": "MLP", "0E": "MLP",
                                                          "XE": "MLP"}}
        df = full_genre_labels(df, replace_dict=replace_dict)
    elif j == "nonR-vs-R":
        cat_labels = ["N", "E", "0E", "XE", "R"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(axis=1)]
        replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "nonR", "E": "nonR", "0E": "nonR",
                                                          "XE": "nonR", "M":"nonR"}}
        df = full_genre_labels(df, replace_dict=replace_dict)

    elif j == "N-vs-E":
        cat_labels = ["N", "E"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(axis=1)]

    if k == "all_media":
        cat_labels = ["Taschenbuch", "Zeitschrift", "Zeitung", "Rundschau", "Anthologie", "Familienblatt", "Sammlung", "Werke","Buch"]
        df = df[df.isin({"Medientyp_ED": cat_labels}).any(axis=1)]
        replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte":"Journal",
                                         "Werke": "Buch", "Nachlass": "Buch", "Kalender":"Taschenbuch",
                                          "Zyklus":"Anthologie", "Sammlung":"Anthologie"}}
        df = full_genre_labels(df, replace_dict=replace_dict)

    elif k == "TB-Ant-RS-FB":
        cat_labels = ["Taschenbuch", "Rundschau", "Anthologie", "Familienblatt"]
        df = df[df.isin({"Medientyp_ED": cat_labels}).any(axis=1)]
    elif k == "TB-Ant-RS-FB-Journal":
        cat_labels = ["Taschenbuch", "Rundschau", "Anthologie", "Familienblatt", "Zeitung", "Zeitschrift"]
        df = df[df.isin({"Medientyp_ED": cat_labels}).any(axis=1)]
        replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal"}}
        df = full_genre_labels(df, replace_dict=replace_dict)

    elif k == "TB-Journal-Book":
        #cat_labels = ["Taschenbuch", "Rundschau", "Anthologie", "Familienblatt", "Zeitung", "Zeitschrift", "Buch", "Werke"]
        #df = df[df.isin({"Medientyp_ED": cat_labels}).any(1)]
        replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal","Illustrierte": "Journal",
                                         "Rundschau": "Journal", "Familienblatt":"Journal",
                                         "Werke":"Buch", "Nachlass": "Buch",
                                         "Zyklus": "Anthologie", "Sammlung": "Anthologie"
                                         }}
        df = full_genre_labels(df, replace_dict=replace_dict)
    elif k == "Buch-vs-NichtBuch":
        # cat_labels = ["Taschenbuch", "Rundschau", "Anthologie", "Familienblatt", "Zeitung", "Zeitschrift", "Buch", "Werke"]
        # df = df[df.isin({"Medientyp_ED": cat_labels}).any(1)]
        replace_dict = {"Medientyp_ED": {"Zeitung": "NichtBuch", "Zeitschrift": "NichtBuch",
                                         "Illustrierte": "NichtBuch",
                                         "Rundschau": "NichtBuch", "Familienblatt": "NichtBuch",
                                         "Taschenbuch":"NichtBuch",
                                         "Werke": "NichtBuch", "Nachlass": "NichtBuch", "Journal":"NichtBuch",
                                         "Kalender":"NichtBuch",
                                         "Zyklus": "NichtBuch", "Sammlung": "NichtBuch", "Anthologie": "NichtBuch"
                                         }}

        df = full_genre_labels(df, replace_dict=replace_dict)

    CrosstabResult = pd.crosstab(index=df['Gattungslabel_ED_normalisiert'],
                                 columns=df['Medientyp_ED'])
    print(j,k)
    print(CrosstabResult)

    # Performing Chi-sq test
    ChiSqResult = chi2_contingency(CrosstabResult)

    # P-Value is the Probability of H0 being True
    # If P-Value&gt;0.05 then only we Accept the assumption(H0)

    print('The P-Value of the ChiSq Test is for ', j," / " , k,  ChiSqResult[1])

