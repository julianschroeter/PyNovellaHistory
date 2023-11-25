from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
import pandas as pd
from numpy import mean, std

from classification.perspectivalmodeling import split_features_labels
from preprocessing.sampling import equal_sample


def resample_boostrapped_LR(n, df, genre_category="Gattungslabel_ED_normalisiert",genre_labels=["N", "E"], train_size=0.8):
    accuracy_scores_list, f1_scores_list = [], []

    df_1 = df[df[genre_category] == genre_labels[0]]
    df_2 = df[df[genre_category] == genre_labels[1]]

    for i in range(n):
        sample = equal_sample(df_1, df_2, minor_frac=1.0)

        X, Y = split_features_labels(sample)
        #X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
         #                                                                               train_size=train_size,
          #                                                                              random_state=42)
        model = LogisticRegression(solver='liblinear', multi_class='auto')

        #model.fit(X_train, Y_train)
        #fit_predictions = model.predict(X_validation)
        #accuracy_scores_list.append(accuracy_score(Y_validation, fit_predictions))
        #f1_scores_list.append(f1_score(Y_validation, fit_predictions, pos_label=genre_labels[0]))
        scores = cross_val_score(model, X, Y, cv=3)
        for element in scores:
            accuracy_scores_list.append(element)
    return mean(accuracy_scores_list), std(accuracy_scores_list)

def resample_boostrapped_SVM(n, df, genre_category="Gattungslabel_ED_normalisiert",genre_labels=["N", "E"], train_size=0.8):
    accuracy_scores_list, f1_scores_list = [], []

    df_1 = df[df[genre_category] == genre_labels[0]]
    df_2 = df[df[genre_category] == genre_labels[1]]

    for i in range(n):
        sample = equal_sample(df_1, df_2, minor_frac=1.0)
        X, Y = split_features_labels(sample)
        #X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
         #                                                                               train_size=train_size,
          #                                                                              random_state=42)
        model = svm.SVC(kernel="linear", C=1.0, random_state=42)
        scores = cross_val_score(model, X, Y, cv=5)


        for element in scores:
            accuracy_scores_list.append(element)


    return mean(accuracy_scores_list), std(accuracy_scores_list)



def logreg_cv(df, genre_category="Gattungslabel_ED_normalisiert",genre_labels=["N", "E"], train_size=0.8):
    accuracy_scores_list, f1_scores_list = [], []

    X, Y = split_features_labels(df)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,
                                                                                        train_size=train_size,
                                                                                        random_state=42)
    lr_model = LogisticRegressionCV(cv=10, solver='liblinear', multi_class="auto")
    lr_model.fit(X_train, Y_train)
    test_predictions = lr_model.predict(X_test)

    return lr_model.score(X,Y), accuracy_score(Y_test, test_predictions), classification_report(Y_test, test_predictions)