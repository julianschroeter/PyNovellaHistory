system = "my_xps" #  "wcph113"

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import os
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing.presetting import global_corpus_representation_directory
from classification.custom_classification import resample_boostrapped_LR
import seaborn as sns
from preprocessing.sampling import equal_sample
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

metadata_filepath = metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
df = pd.read_csv(metadata_filepath, index_col=0)

df = df[["Gattungslabel_ED_normalisiert", "Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED"]]


values_dict = {"Gattungslabel_ED_normalisiert": ["N", "E"]}
df = df[df.isin(values_dict).any(axis=1)]
df = df.dropna()

df_N = df[df["Gattungslabel_ED_normalisiert"] == "N"]
df_E = df[df["Gattungslabel_ED_normalisiert"] == "E"]

lin_scores, nonlin_scores, lin_scores_opt, nonlin_scores_opt = [],[], [],[]
n = 100
for i in range(n):

    sample = equal_sample(df_N, df_E)


    labels = sample["Gattungslabel_ED_normalisiert"]
    data =  sample[["Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED"]]
    columns_list = ["Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED"]


    df_dummies = pd.get_dummies(data, columns=columns_list)

    X = df_dummies.values
    Y = labels.values

    train_size = 0.80

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=train_size)

    df_dummies = pd.concat([df_dummies, labels], axis=1)


    # Train a linear SVM
    linear_svm = SVC(kernel='linear', C=1)
    linear_svm.fit(X_train, y_train)
    y_pred_linear = linear_svm.predict(X_test)
    print("Linear SVM Classification Report:")
    print(classification_report(y_test, y_pred_linear))
    print("Accuracy: ", accuracy_score(y_test, y_pred_linear))
    lin_scores.append(accuracy_score(y_test, y_pred_linear))

    # plot heatmap with confusion matrix
    #cm_linear = confusion_matrix(y_test, y_pred_linear)
    #plt.figure(figsize=(8, 4))
    #sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', xticklabels="", yticklabels="")
    #plt.title('Confusion Matrix - Linear SVM')
    #plt.xlabel('Predicted')
    #plt.ylabel('True')
    #plt.show()

    # Train a non-linear SVM with RBF kernel
    nonlinear_svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))
    nonlinear_svm.fit(X_train, y_train)
    y_pred = nonlinear_svm.predict(X_test)
    print("Non-linear SVM Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    nonlin_scores.append(accuracy_score(y_test, y_pred))



    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    kernel = ['rbf']
    param_grid = dict(kernel=kernel, gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X_train, y_train)

    print(
        "The best parameters for a linear SVM are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )
    print(grid.best_params_['C'])
    print(grid.best_params_['gamma'])
    print(grid.best_params_['kernel'])

    c = grid.best_params_['C']
    gamma = grid.best_params_['gamma']


    # Train a linear SVM
    linear_svm = SVC(kernel='linear', C=c, gamma=gamma)
    linear_svm.fit(X_train, y_train)
    y_pred_linear = linear_svm.predict(X_test)
    print("Linear SVM Classification Report:")
    print(classification_report(y_test, y_pred_linear))
    print("Accuracy: ", accuracy_score(y_test, y_pred_linear))
    lin_scores_opt.append(accuracy_score(y_test, y_pred_linear))

    #grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv)
    #grid.fit(X_train, y_train)

    print(
        "The best parameters for a nonlinear SVM are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )
    print(grid.best_params_['C'])
    print(grid.best_params_['gamma'])

    c = grid.best_params_['C']
    gamma = grid.best_params_['gamma']


    # Train a non-linear SVM with RBF kernel
    nonlinear_svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=c, gamma="scale"))
    nonlinear_svm.fit(X_train, y_train)
    y_pred = nonlinear_svm.predict(X_test)
    print("Non-linear SVM Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    nonlin_scores_opt.append(accuracy_score(y_test, y_pred))

print(lin_scores)
print(nonlin_scores)
print(lin_scores_opt)
print(nonlin_scores_opt)

print("Mean:")

print(np.mean(lin_scores))
print(np.mean(nonlin_scores))

print(np.mean(lin_scores_opt))
print(np.mean(nonlin_scores_opt))

print("Std:")

print(np.std(lin_scores))
print(np.std(nonlin_scores))

print(np.std(lin_scores_opt))
print(np.std(nonlin_scores_opt))


print("Improvement Rates without optimization:")
print(np.mean(nonlin_scores) / np.mean(lin_scores))
print("Improvement Rates with optimization:")
print(np.mean(nonlin_scores_opt) / np.mean(lin_scores_opt))


print("Improvement Rates linear optimization over non-optimzation:")
print(np.mean(lin_scores_opt) / np.mean(lin_scores))
print("Improvement Rates nonlinear optimization over nonlinear non-optimization:")
print(np.mean(nonlin_scores_opt) / np.mean(nonlin_scores))