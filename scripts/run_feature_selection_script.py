from sklearn.feature_selection import RFECV
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from SupervisedLearning.PerspectivalModeling import split_features_labels
from preprocessing.presetting import local_temp_directory, global_corpus_raw_dtm_directory, global_corpus_representation_directory
from preprocessing.corpus import DTM
import os
import matplotlib.pyplot as plt
import pandas as pd

system = "wcph113" # "my_mac"

dtm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), "dtm_tfidf_genrelabel.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

dtm_object = DTM(data_matrix_filepath =dtm_infile_path, metadata_csv_filepath=metadata_filepath)

dtm_object = dtm_object.reduce_to_categories(metadata_category="Gattungslabel_ED", label_list=["M", "R", "N", "E", "0E"])

input_df = dtm_object.data_matrix_df
print(input_df)

X, Y = split_features_labels(input_df)
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.1, random_state=42)
print(X)
print(Y)

min_features_to_select = 1
models = [("LR", LogisticRegression(solver="lbfgs", penalty="l2", multi_class="ovr")), ("SVM", SVC(kernel="linear"))]
for name, estimator in models:
    selector = RFECV(estimator, min_features_to_select=min_features_to_select, step=1, cv=3)

    selector.fit(X, Y)
    supported_features = selector.support_
    supported_features = supported_features.tolist()

    dropped_input_df = input_df.drop(columns=["Gattungslabel_ED"])
    new_df = dropped_input_df.loc[:, supported_features]
    print(new_df)
    new_df_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), str("RFECV_reduced_dtm" + name + ".csv"))
    fig_filename = os.path.join(local_temp_directory(system), str("RFE_plot" + name + ".png"))
    new_df.to_csv(path_or_buf= new_df_outfile_path)
    print(str(name) + ": Optimal number of features : %d" % selector.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()

    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(range(min_features_to_select,
               len(selector.grid_scores_) + min_features_to_select),
         selector.grid_scores_)
    plt.show()
    plt.savefig(fig_filename)