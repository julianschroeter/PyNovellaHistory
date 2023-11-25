system = "my_xps" # "wcph113" #   "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os
import numpy as np
from numpy import unique
from numpy import where

from pprint import pprint  # Look at parameters used by our current forest

from preprocessing.presetting import global_corpus_representation_directory, global_corpus_directory, language_model_path, vocab_lists_dicts_directory, word_translate_table_to_dict, global_corpus_raw_dtm_directory, local_temp_directory
from preprocessing.corpus import DTM
from preprocessing.sampling import sample_n_from_cat
from classification.perspectivalmodeling import split_features_labels
from clustering.my_plots import plot_prototype_concepts
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]  # Create the random grid
random_grid = {'n_estimators': n_estimators,
               #  'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)


metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

label_list = ["R", "E"]

filename = "red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" # "raw_dtm_l1_lemmatized_use_idf_False2500mfw.csv" #    "no-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" #
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)
dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)

dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED", "Nachname"])
dtm_obj = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=label_list)
dtm_obj = dtm_obj.eliminate(["roman", "märchen", "novelle", "erzählung","fle", "be", "te", "ge"])
year_labels = dtm_obj.data_matrix_df["Jahr_ED"].to_list()
#dtm_obj = dtm_obj.eliminate(["Jahr_ED"])

df_init = dtm_obj.data_matrix_df
df_init = df_init[df_init["Jahr_ED"] >= 1850]


df_init = df_init.drop(columns=["Jahr_ED"])
indexes = df_init.index.tolist()
features = df_init.columns.tolist()



rf_acc_scores, lr_acc_scores = [],[]
n=10
for i in range(n):
    df = sample_n_from_cat(df_init)

    X, Y_orig = split_features_labels(df)

    subs_dict = {"E": 1, "R": 0}
    Y = list(map(subs_dict.get, Y_orig, Y_orig))

    subs_dict = {1: "green", 0: "blue"}
    Y_color = list(map(subs_dict.get, Y, Y))

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8)

    rf_model = RandomForestClassifier()
    lr_model = LogisticRegressionCV()

    rnd_model = RandomForestClassifier()
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8)

    rf_random = RandomizedSearchCV(estimator=rnd_model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)  # Fit the random search model
    rf_random.fit(X_train, Y_train)
    test_predictions = rf_random.best_estimator_.predict(X_test)

    print(rf_random.best_params_)
    print(accuracy_score(Y_test, test_predictions))

#    rf_model.fit(X_train, Y_train)
    lr_model.fit(X_train, Y_train)

    rf_test_predictions = rf_random.best_estimator_.predict(X_test)

    rf_acc = accuracy_score(Y_test, rf_test_predictions)
    rf_acc_scores.append(rf_acc)

    lr_test_predictions = lr_model.predict(X_test)
    lr_acc = accuracy_score(Y_test, lr_test_predictions)
    lr_acc_scores.append(lr_acc)


print(rf_acc_scores,lr_acc_scores)

import numpy as np
lr_mean = np.mean(np.asarray(lr_acc_scores))
print(lr_mean)

rf_mean = np.mean(np.asarray(rf_acc_scores))
print(rf_mean)


from scipy import stats
print(stats.ttest_rel(rf_acc_scores, lr_acc_scores, alternative="greater"))
print("after 1850")
print("Gernes to be compared: ", label_list)

print(filename)
print("iterations: ", n)
print("Improvement: ", rf_mean / lr_mean)




