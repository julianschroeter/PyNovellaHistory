from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.sampling import equal_sample
from preprocessing.metadata_transformation import years_to_periods
from classification.custom_classification import resample_boostrapped_LR
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score, classification_report

system_name =  "wcph113" # "my_mac" # "wcph104" "my_xps" #

sujet_data_matrix_filepath = os.path.join(global_corpus_representation_directory(system_name), "DocThemesMatrix.csv")
NE_data_matrix_filepath = os.path.join(global_corpus_representation_directory(system_name), "toponym_share_Matrix.csv")

language_model = language_model_path(system_name)
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system_name), "Bibliographie.csv")


textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system_name), "textanalytic_metadata.csv")

matrix = DocFeatureMatrix(data_matrix_filepath= sujet_data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = NE_data_matrix_filepath, mallet=False)

matrix = matrix.reduce_to(["Marseille"])
matrix = matrix.add_metadata(["rom_top"])


matrix = DocFeatureMatrix(data_matrix_filepath= None, data_matrix_df=matrix.data_matrix_df, metadata_df=None,
                                  metadata_csv_filepath = textanalytic_metadata_filepath, mallet=False)

matrix = matrix.add_metadata("region")

matrix.data_matrix_df.replace({"region": {"Italien": "1_Romanisch",
                                                        "Spanien": "1_Romanisch",
                                                        "Frankreich": "1_Romanisch", "Lateinamerika" : "3_Lateinam",
                                          "Karibik" : "3_Lateinam", "Chile" : "3_Lateinam", "Portugal":"1_Romanisch",
                                          "Deutschland" : "2_nicht-Rom", "Österreich" : "2_nicht-Rom",
                                          "Niederlande" : "2_nicht-Rom", "Ungarn" : "2_nicht-Rom", "Russlan" : "2_nicht-Rom",
                                          "Polen" : "2_nicht-Rom", "Schweden" : "2_nicht-Rom", "Universum" : "2_nicht-Rom", "Dänemark" :"2_nicht-Rom",
                                          "Russland" : "2_nicht-Rom", "Schweiz" : "2_nicht-Rom", "Karibik" : "3_Lateinam", "Alpen" : "2_nicht-Rom",
                                          "Nordamerika" :"2_nicht-Rom", "Meer":"2_nicht-Rom", "unbestimmt": "4_nicht-annot"

                                          }}, inplace=True)

matrix.data_matrix_df["region"] = matrix.data_matrix_df["region"].fillna("4_nicht-annot", inplace=False)


matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix.data_matrix_df, metadata_csv_filepath=metadata_csv_filepath)

matrix = matrix.add_metadata(("Gattungslabel_ED_normalisiert"))

label_list = ["R", "M", "E", "N", "0E", "XE"]
genre_cat = "Gattungslabel_ED_normalisiert"
matrix = matrix.reduce_to_categories(genre_cat, label_list)

df = matrix.data_matrix_df
df = df.rename(columns={"Marseille":"SujetShare", "rom_top":"NamedEntShare"})


scaler = StandardScaler()
df.iloc[:, :2] = scaler.fit_transform(df.iloc[:, :2].to_numpy())


df_rom = df[df["region"] == "1_Romanisch"]
print("statistics für roman setting and genre: ", len(df_rom))
print(df_rom.groupby(["Gattungslabel_ED_normalisiert"]).count())


df_non_rom = df[df["region"] == "2_nicht-Rom"]
print("statistics for non-roman setting and genre: ", len(df_non_rom))
print(df_non_rom.groupby(["Gattungslabel_ED_normalisiert"]).count())


category = genre_cat# "liebesspannung" # "stadt_land" #
first_value ="N" # "ja" #"land" #
second_value = "E" #"nein" # "None" #≈

df_boxpl = df
df_boxpl = df_boxpl.query("region != '''4_nicht-annot'''")
df_boxpl = df_boxpl.query("region != '''3_Lateinam'''")

df_boxpl.boxplot(by=category)
df_boxpl.boxplot(by="region")
plt.show()


df_class = df.drop(columns=["region", "SujetShare"])

sample_df = df_class[df_class[category] == first_value]
counter_sample_df = df_class[df_class[category] == second_value]

class_df = equal_sample(sample_df, counter_sample_df)
all_class_df = pd.concat([sample_df, counter_sample_df]).sample(frac=1)
class_df = class_df.sample(frac=1)

sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)



# classification

lr_model = LogisticRegressionCV()
dt_clf = DecisionTreeClassifier(max_leaf_nodes=2)

array = all_class_df.to_numpy()
X = array[:, 0:(array.shape[1]-1)]
Y = array[:, array.shape[1]-1]


X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2,random_state=2)
lr_model.fit(X_train, Y_train)
dt_clf.fit(X_train, Y_train)
predictions = lr_model.predict(X_validation)
print("Accuracy score: ", accuracy_score(Y_validation, predictions))
print("cv score: ", lr_model.score(X, Y))
print(classification_report(Y_validation, predictions))
print("coef:" , lr_model.coef_)
print("Schwellenwert: " , 0.5 / lr_model.coef_)


means, stds = resample_boostrapped_LR(n=100, df=df_class, genre_category=genre_cat, genre_labels=["N", "E"] )
print("mean bootstr acc score: ", np.array(means).mean())

year_cat = "Jahr_ED"
print("Calculate classification report for temporalized analysis: after 1850:")

matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix.data_matrix_df, metadata_csv_filepath=metadata_csv_filepath)

matrix = matrix.add_metadata(year_cat)
matrix.data_matrix_df = years_to_periods(matrix.data_matrix_df,category_name=year_cat, start_year=1750, end_year=1950, epoch_length=100,
                                          new_periods_column_name="periods")

df = matrix.data_matrix_df
after_1850_df = df[df["periods"] == "1850-1950"]
print(after_1850_df)
df = after_1850_df
df = df.rename(columns={"Marseille":"SujetShare", "rom_top":"NamedEntShare"})

df = df.drop(columns=["periods", year_cat, "SujetShare"])

print(df)

df.boxplot(by=category)
plt.show()


df_class = df.drop(columns=["region"])

sample_df = df_class[df_class[category] == first_value]
counter_sample_df = df_class[df_class[category] == second_value]

import pandas as pd
class_df = equal_sample(sample_df, counter_sample_df)
class_df = class_df.sample(frac=1)
all_class_df = pd.concat([sample_df, counter_sample_df]).sample(frac=1)

sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)



# classification

lr_model = LogisticRegressionCV()
dt_clf = DecisionTreeClassifier(max_leaf_nodes=2)

array = all_class_df.to_numpy()
X = array[:, 0:(array.shape[1]-1)]
Y = array[:, array.shape[1]-1]


X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2,random_state=2)
print(X_train)
print(Y_train)
lr_model.fit(X_train, Y_train)
dt_clf.fit(X_train, Y_train)
predictions = lr_model.predict(X_validation)
print("Accuracy score: ", accuracy_score(Y_validation, predictions))
print("cv score: ", lr_model.score(X, Y))
print(classification_report(Y_validation, predictions))
print("coef:" , lr_model.coef_)
print("Schwellenwert: " , 0.5 / lr_model.coef_)

means, stds = resample_boostrapped_LR(n=100, df=df_class, genre_category=genre_cat, genre_labels=["N", "E"] )
print("mean bootstr acc score: ", np.array(means).mean())
