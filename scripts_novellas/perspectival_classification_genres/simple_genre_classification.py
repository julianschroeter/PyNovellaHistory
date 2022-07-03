system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os

from preprocessing.presetting import global_corpus_representation_directory, global_corpus_directory, language_model_path, vocab_lists_dicts_directory, word_translate_table_to_dict, global_corpus_raw_dtm_directory
from preprocessing.corpus import DTM
from classification.perspectivalmodeling import split_features_labels
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from preprocessing.sampling import equal_sample

metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

label_list = ["N", "E"]

for filename in os.listdir(global_corpus_raw_dtm_directory(system)):
    if filename == "red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv":
        filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)
        dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)

        dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert"])
        dtm_obj = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=label_list)
        dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge"])

        df = dtm_obj.data_matrix_df

        X, Y = split_features_labels(df)
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.5,  random_state=42)
        lr_model = LogisticRegressionCV(cv=10, solver='liblinear', multi_class="auto")
        lr_model.fit(X_train, Y_train)
        test_predictions = lr_model.predict(X_test)

        coef = lr_model.coef_.tolist()
        coef = [item for sublist in coef for item in sublist]
        df = df.drop(columns=["Gattungslabel_ED_normalisiert"])
        features = df.columns.tolist()

        print("classification report with inverted train/test sets for " + str(filename) + ": ")
        print("CV based score: ", lr_model.score(X, Y))
        print("accuracy score on test set: ", accuracy_score(Y_test, test_predictions))
        print(classification_report(Y_test, test_predictions))

        print(confusion_matrix(Y_test, test_predictions))