system = "my_xps" #  "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import stats
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_directory, language_model_path, vocab_lists_dicts_directory, word_translate_table_to_dict, global_corpus_raw_dtm_directory, local_temp_directory
from preprocessing.corpus import DTM
from classification.perspectivalmodeling import split_features_labels
from sklearn import model_selection
from sklearn.svm import SVC, LinearSVC
from preprocessing.sampling import equal_sample, principled_sampling
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pandas as pd

name_cat = "Nachname"
periods_cat = "Jahr_ED"
genre_cat = "Gattungslabel_ED_normalisiert"

metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
label_list = ["R", "E"]

filename = "scaled_raw_dtm_l1__use_idf_False2500mfw.csv" # "red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" #"raw_dtm_l1_lemmatized_use_idf_False2500mfw.csv" # "no-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv":
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)
dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)

lin_scores, nonlin_scores, lin_scores_opt, nonlin_scores_opt = [],[], [],[]
for i in range(10):


    dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)
    dtm_obj.data_matrix_df = dtm_obj.data_matrix_df.sample(frac=0.9, axis="columns")

    dtm_obj = dtm_obj.add_metadata([genre_cat, name_cat, periods_cat])
    dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=label_list)
    dtm_obj = dtm_obj.eliminate(["roman", "märchen", "fle", "be", "te", "ge"])


    df = dtm_obj.data_matrix_df
    df_0 = df[df[genre_cat] == label_list[0]]
    df_1 = df[df[genre_cat] == label_list[1]]

    train_set, test_set = principled_sampling(df_1, df_0)

    indexes = df.index.tolist()
    features = df.columns.tolist()

    X_train, Y_train_orig = split_features_labels(train_set)
    X_test, Y_test_orig = split_features_labels(test_set)

    subs_dict = {label_list[0]: 1, label_list[1]: 0}
    y_train = list(map(subs_dict.get, Y_train_orig, Y_train_orig))
    y_test = list(map(subs_dict.get, Y_test_orig, Y_test_orig))

    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=0.8)

    lin_model = LinearSVC()
    nonlin_model = SVC()

    # train a linear SVM :
    lin_model.fit(X_train, y_train)
    y_pred_linear = lin_model.predict(X_test)
    print("Linear SVM Classification Report:")
    print(classification_report(y_test, y_pred_linear))
    print("Accuracy: ", accuracy_score(y_test, y_pred_linear))
    lin_scores.append(accuracy_score(y_test, y_pred_linear))

    # Train a non-linear SVM with RBF kernel and standard parameter settings
    nonlinear_svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))
    nonlinear_svm.fit(X_train, y_train)
    y_pred = nonlinear_svm.predict(X_test)
    print("Non-linear SVM Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    nonlin_scores.append(accuracy_score(y_test, y_pred))

    # do a grid search for optimal parameters C and gamma for non-linear rbf kernel SVM:
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

    # Train a non-linear SVM with RBF kernel with optimal parameters:
    nonlinear_svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=c, gamma="scale"))
    nonlinear_svm.fit(X_train, y_train)
    y_pred = nonlinear_svm.predict(X_test)
    print("Non-linear SVM Classification Report after optimization:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    nonlin_scores_opt.append(accuracy_score(y_test, y_pred))


print("Genres to be compared: ", label_list)
print("all scores (lin, non-lin, optimized non-lin): ")
print(lin_scores, nonlin_scores, nonlin_scores_opt)

lin_mean = np.mean(np.asarray(lin_scores))
print("Mean accuracy for linear SVM: ", lin_mean)

nonlin_mean = np.mean(np.asarray(nonlin_scores))
print("Mean accuracy for non-linear SVM: ", nonlin_mean)

nonlin_opt_mean = np.mean(np.asarray(nonlin_scores_opt))
print("Mean accuracy for (C and gamma) optimized nonlinear SVM: ", nonlin_opt_mean)

print("T-Test for nonlin vs lin:")
print(stats.ttest_rel(nonlin_scores, lin_scores, alternative="greater"))

print("T-Test for optimized nonlin vs lin:")
print(stats.ttest_rel(nonlin_scores, lin_scores, alternative="greater"))

print("Improvement non-lin vs. lin svm model: ", nonlin_mean / lin_mean)
print("Improvement optimized non-lin vs. lin svm model: ", nonlin_opt_mean / lin_mean)