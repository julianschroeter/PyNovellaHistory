system = "my_xps" #  "wcph113"

import pandas as pd
import numpy as np
import os
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing.presetting import global_corpus_representation_directory
from classification.custom_classification import resample_boostrapped_LR

metadata_filepath = metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
df = pd.read_csv(metadata_filepath, index_col=0)

df = df[["Gattungslabel_ED_normalisiert", "Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED"]]

genres_list = ["E", "R"]
print(genres_list)
values_dict = {"Gattungslabel_ED_normalisiert": genres_list}
df = df[df.isin(values_dict).any(axis=1)]
df = df.dropna()

labels = df["Gattungslabel_ED_normalisiert"]
data =  df[["Nachname"]] # [,"Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED",  "Jahr_ED"]]
columns_list = ["Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED"]
columns_list = ["Nachname"] #", "Jahr_ED"
print("factors: ", columns_list)
df_dummies = pd.get_dummies(data, columns=columns_list)

X = df_dummies.values
Y = labels.values

train_size = 0.80
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, train_size=train_size, random_state=seed)

print(len(Y_train), len(Y_validation))

model = LogisticRegression(solver="liblinear", multi_class="ovr")
model.fit(X_train, Y_train)

predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


print("same process based on bootstrapped resampling with equal sample size:")

df_dummies = pd.concat([df_dummies, labels], axis=1)

print(df_dummies)


acc, std = resample_boostrapped_LR(n=100, df=df_dummies, genre_category="Gattungslabel_ED_normalisiert",genre_labels=genres_list, train_size=0.8)

print("accuracy score results (all results, mean, std):")
print(acc)
print(std)


