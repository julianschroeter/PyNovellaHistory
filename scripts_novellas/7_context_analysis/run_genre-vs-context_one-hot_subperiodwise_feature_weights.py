system = "my_xps" #  "wcph113"

import pandas as pd
import numpy as np
import os
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from preprocessing.presetting import global_corpus_representation_directory
from preprocessing.metadata_transformation import years_to_periods, full_genre_labels
from preprocessing.sampling import equal_sample, split_to2samples
from classification.custom_classification import resample_boostrapped_LR
from collections import defaultdict

metadata_filepath = metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
df = pd.read_csv(metadata_filepath, index_col=0)

df = df[["Gattungslabel_ED_normalisiert", "Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED"]]


labels = ["N", "R"]
labels = ["N", "R"]
labels = ["N", "E", "0E", "XE", "R"]
values_dict = {"Gattungslabel_ED_normalisiert": labels}
df = df[df.isin(values_dict).any(axis=1)]
df = df.dropna()

medium_cat = "Medientyp_ED"
df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch", "(unbekannt)":"Buch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)


subs_dict = {"N": 1, "R": 0}
subs_dict = {"N": 1, "E": 1, "XE":1, "0E":1, "R":0}
df = df.replace({"Gattungslabel_ED_normalisiert":subs_dict})

data = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1770, end_year=1950, epoch_length=5,
                      new_periods_column_name="periods5a")


data =  data[["Nachname" , "periods5a" , "Gender","Jahr_ED", "Gattungslabel_ED_normalisiert"]] #   "Kanon_Status", "Medientyp_ED" ,
data = years_to_periods(input_df=data, category_name="Jahr_ED", start_year=1770, end_year=1910, epoch_length=40,
                      new_periods_column_name="periods")



columns_list = ["periods5a", "Nachname"  , "Gender" ] # "Jahr_ED" ,,  ,"Kanon_Status",  "Medientyp_ED" ,

train_size = 0.8

periods = list(set(data.periods.values.tolist()))
print(periods)
periods = [x for x in periods if x != 0]
periods.sort()
for period in periods:
    print("period is: ", period)
    period_data = data[data["periods"] == period]



    n = 5
    lr_acc_scores, N_f1_scores = [], []
    d = defaultdict(list)
    for i in range(n):
         #
        df1, df2 = split_to2samples(period_data, "Gattungslabel_ED_normalisiert", label_list= [0,1])
        sample = equal_sample(df1, df2)
        labels = sample["Gattungslabel_ED_normalisiert"]
        sample = sample.drop(columns=["periods", "Gattungslabel_ED_normalisiert", "Jahr_ED"])
        df_dummies = pd.get_dummies(sample, columns=columns_list)

        X = df_dummies.values
        Y = labels.values
        print("sample size: ", len(X), len(Y))
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, train_size=train_size)

        model = LogisticRegression(solver="liblinear", multi_class="ovr")
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        lr_acc_scores.append(accuracy_score(Y_validation, predictions))
        N_f1_scores.append(f1_score(Y_validation, predictions))
        features = df_dummies.columns.tolist()
        coefficients = model.coef_.tolist()
        coefficients = [item for sublist in coefficients for item in sublist]

        feat_coef = list(zip(features, coefficients))

        for k, v in feat_coef:
            d[k].append(v)

    for k,v in d.items():
        d[k] = [np.mean(np.array(v)), np.std(np.array(v))]

    print(d)

    coef_df = pd.DataFrame.from_dict(d, orient="index", columns=["mean", "st_dev"])
    coef_df = coef_df.sort_values(by="mean")
    print(coef_df)

    print("Len train, test sets: ", len(Y_train), len(Y_validation))
    print("mean of accuracy scores for LR:")
    print(np.mean(np.array(lr_acc_scores)))
    print("mean of f1 scores for LR for Novelle:")
    print(np.mean(np.array(N_f1_scores)))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))



