from preprocessing.presetting import language_model_path, local_temp_directory, vocab_lists_dicts_directory, global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory
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

system = "my_xps" # "wcph113" # "my_mac" # "wcph104"

infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
language_model = language_model_path(system)
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system), "Bibliographie.csv")


textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "textanalytic_metadata.csv")

matrix = DocFeatureMatrix(data_matrix_filepath= infile_df_path, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = metadata_csv_filepath, mallet=False)
matrix = matrix.reduce_to(["token_count"])
matrix = matrix.add_metadata(("Gattungslabel_ED_normalisiert"))

label_list = ["R", "M", "E", "N", "0E", "XE"]
label_list = ["R", "N"]
genre_cat = "Gattungslabel_ED_normalisiert"
matrix = matrix.reduce_to_categories(genre_cat, label_list)


df = matrix.data_matrix_df


scaler = StandardScaler()
#df.iloc[:, :2] = scaler.fit_transform(df.iloc[:, :2].to_numpy())


first_value ="N" # "ja" #"land" #
second_value = "R" #"nein" # "None" #≈

# classification

lr_model = LogisticRegressionCV()

array = df.to_numpy()
X = array[:, 0:(array.shape[1]-1)]
Y = array[:, array.shape[1]-1]

x_boundaries = []
n = 100
for i in range(n):
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.3)
    lr_model.fit(X_train, Y_train)
    predictions = lr_model.predict(X_validation)
    print("Accuracy score: ", accuracy_score(Y_validation, predictions))
    print("cv score: ", lr_model.score(X, Y))
    print(classification_report(Y_validation, predictions))
    print("coef:" , lr_model.coef_)
    print("Schwellenwert: " , lr_model.coef_.T)
    print("Y-Abschnitt: " , lr_model.intercept_)

    lengths = np.arange(20000, 80000)
    predictions = lr_model.predict(lengths.reshape(-1, 1))
    probabs = lr_model.predict_proba(lengths.reshape(-1,1))
    probabs = [x[1] for x in probabs]
    pred_colors = ["red" if x == "N" else "blue" for x in predictions]

    prob_df = pd.DataFrame(data= {"lengths": lengths, "probabs": probabs,
                                  "predictions": predictions})
    print(prob_df)

    x_boundary = prob_df[prob_df["probabs"] >= 0.5].iloc[0, 0]
    x_boundaries.append(x_boundary)

x_boundaries = np.array(x_boundaries)

print(np.mean(x_boundaries))
print(np.std(x_boundaries))

print(np.percentile(x_boundaries, [5,95]))
left, right = np.percentile(x_boundaries, [5]), np.percentile(x_boundaries, [95])

import matplotlib.pyplot as plt
plt.scatter(lengths, probabs, c=pred_colors)
plt.title("Entscheidungsgrenze für die Klassifikation: Novelle vs. Roman")
plt.ylabel("Vorhersagewahrscheinlichkeit")
plt.xlabel("Textlänge")
plt.vlines(left, 0, 1, label="Decision boundary")
plt.vlines(right, 0, 1, label="Decision boundary")
#plt.hlines(0.5, 20000, np.mean(np.array(x_boundaries)), label="Decision Boundary: ")
plt.text(right, 0, "Entscheidungsgrenzbereich um: " + str(int(np.mean(np.array(x_boundaries)))), ha='left', va='center')
plt.legend
plt.savefig(os.path.join(local_temp_directory(system), "figures", "Abb_Entscheidungsgrenze_sigmoid_function_length.svg"))
plt.show()

print( 50000 * lr_model.coef_ + lr_model.intercept_)

