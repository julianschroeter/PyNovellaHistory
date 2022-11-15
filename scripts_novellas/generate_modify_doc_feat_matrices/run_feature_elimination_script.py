system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from classification.perspectivalmodeling import split_features_labels
from preprocessing.presetting import local_temp_directory, global_corpus_raw_dtm_directory, global_corpus_representation_directory
from preprocessing.corpus_alt import DTM
import os
import matplotlib.pyplot as plt

infile_name = "no-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
dtm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), infile_name )
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

dtm_object = DTM(data_matrix_filepath =dtm_infile_path, metadata_csv_filepath=metadata_filepath)
dtm_object = dtm_object.add_metadata(["Gattungslabel_ED_normalisiert"])

dtm_object = dtm_object.eliminate(["novelle", "erzählung", "roman", "märchen", "ge", "te", "be"])

df_all = dtm_object.data_matrix_df

label_list = ["M", "R", "N", "E", "0E", "XE"]
dtm_object = dtm_object.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=label_list)
input_df = dtm_object.data_matrix_df

X, Y = split_features_labels(input_df)
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.5, random_state=42)


min_features_to_select = 10
models = [("LR", LogisticRegression(solver="liblinear", penalty="l2", multi_class="auto")), ("SVM", SVC(kernel="linear"))]
for name, estimator in models:
    #selector = RFE(estimator, step=2, n_features_to_select=n_features_to_select)
    selector = RFECV(estimator, min_features_to_select=min_features_to_select)
    selector.fit(X_train, Y_train)
    supported_features = selector.support_
    supported_features = supported_features.tolist()
    print(supported_features)

    dropped_df_all = df_all.drop(columns=["Gattungslabel_ED_normalisiert"])
    new_df = dropped_df_all.loc[:, supported_features]
    print(new_df)
    labels_str = "-".join(label_list)
    new_df_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), str("RFECV_red-to-"+ str(selector.n_features_) + "_" + str(name)+ labels_str + infile_name))
    new_df.to_csv(path_or_buf= new_df_outfile_path)
    print(str(name) + ": Optimal number of features : %d" % selector.n_features_)
    print("dict of cv results: ", selector.cv_results_)


    # fit new model:
    Y, Y = split_features_labels(new_df)
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.5,
                                                                                    random_state=42)
    clf = estimator
    clf.fit(X_train, Y_train)
    test_predictions = clf.predict(X_validation)
    print(classification_report(Y_validation, test_predictions))

    # reverse: train on the rest and validate on the initial training set:
    X_validation, X_train, Y_validation, Y_train = model_selection.train_test_split(X, Y, test_size=0.5,
                                                                                    random_state=42)
    clf = estimator
    clf.fit(X_train, Y_train)
    test_predictions = clf.predict(X_validation)
    print("reversed classification on the complementary sample:")
    print(classification_report(Y_validation, test_predictions))


    # Plot number of features VS. cross-validation scores
    plt.figure()

    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(range(min_features_to_select,
               len(selector.grid_scores_) + min_features_to_select),
         selector.grid_scores_)
    plt.show()

    fig_filename = os.path.join(local_temp_directory(system), str("RFECV_plot" + str(selector.n_features_) + "_" + str(name)+ labels_str +".png"))
    plt.savefig(fig_filename)
