system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os
import numpy as np

from preprocessing.presetting import global_corpus_representation_directory, global_corpus_directory, language_model_path, vocab_lists_dicts_directory, word_translate_table_to_dict, global_corpus_raw_dtm_directory, local_temp_directory
from preprocessing.corpus import DTM
from classification.PerspectivalModeling import split_features_labels
from clustering.my_plots import plot_prototype_concepts
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd

from metrics.scores import c_at_1

metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

label_list = ["N", "E"]

for filename in os.listdir(global_corpus_raw_dtm_directory(system)):
    if "no-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" in filename:
        filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)
        dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)

        dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED"])
        dtm_obj = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=label_list)
        dtm_obj = dtm_obj.eliminate(["roman", "märchen", "fle", "be", "te", "ge"])
        year_labels = dtm_obj.data_matrix_df["Jahr_ED"].to_list()
        dtm_obj = dtm_obj.eliminate(["Jahr_ED"])

        df = dtm_obj.data_matrix_df
        indexes = df.index.tolist()
        features = df.columns.tolist()

        X, Y_orig = split_features_labels(df)

        subs_dict = {"N": 1, "E": 0}
        Y = list(map(subs_dict.get, Y_orig, Y_orig))

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.5, random_state=42)
        rnd_model = RandomForestClassifier()
        rnd_model.fit(X_train, Y_train)
        test_predictions = rnd_model.predict(X_test)

        feat_importance = rnd_model.feature_importances_.tolist()


        importance_features = list(zip(feat_importance, features))
        print("importances with feature names: ", sorted(importance_features, key=lambda x: x[0]))

        print("classification report with regular train/test sets for " + str(filename) + ": ")
        print(rnd_model.score(X, Y))
        print(accuracy_score(Y_test, test_predictions))
        print(classification_report(Y_test, test_predictions))
        predict_probs_inv = [x for sublist in rnd_model.predict_proba(X_test)[:, 0:1] for x in sublist]
        predict_probs = [x for sublist in rnd_model.predict_proba(X)[:, 1:2] for x in sublist]
        fig, ax = plt.subplots()
        subs_dict = {1: "red", 0: "green"}
        genre_c_labels = list(map(subs_dict.get, Y, Y))
        genre_c_labels_test = list(map(subs_dict.get, Y_test, Y_test))
        ax.scatter(year_labels, predict_probs, c=genre_c_labels)
        plt.show()

        threshold_ranges = np.arange(0.01, 1, 0.01)
        thresholds_scores = []
        for threshold in threshold_ranges:
            scores = (threshold, c_at_1(Y_test, predict_probs, threshold))
            thresholds_scores.append(scores)
        scores = [scores[1] for scores in thresholds_scores]

        optimum = max(thresholds_scores, key=lambda scores: scores[1])
        print(optimum)

        plt.plot(np.arange(0.01, 1, 0.01), scores)
        plt.axvline(x=optimum[0])
        plt.axhline(y=optimum[1])
        plt.show()

        plot_prototype_concepts(predict_probs_inv, genre_c_labels_test, threshold=optimum[0])

        predict_probs = [x for sublist in rnd_model.predict_proba(X)[:, 1:2] for x in sublist]
        proba_df = pd.DataFrame(predict_probs, index=indexes, columns=["predict_probab"])
        #proba_df["genre_colors"] = genre_c_labels
        print(proba_df)
        proba_obj = DTM(data_matrix_df=proba_df, metadata_csv_filepath=metadata_path)
        proba_obj = proba_obj.add_metadata(["Gattungslabel_ED_normalisiert","Nachname", "Jahr_ED", "Titel"])
        print(proba_obj.data_matrix_df)
        df = proba_obj.data_matrix_df
        print(df.loc["00306-00", "predict_probab"])
        proba_df_E = df[df["Gattungslabel_ED_normalisiert"] == "E"]
        print(proba_df_E.sort_values(by="predict_probab"))
        proba_df_E.to_csv(path_or_buf=os.path.join(local_temp_directory(system), "Pred_Probab_E.csv"))

        proba_df_N = df[df["Gattungslabel_ED_normalisiert"] == "N"]
        print(proba_df_N.sort_values(by="predict_probab"))
        proba_df_N.to_csv(path_or_buf=os.path.join(local_temp_directory(system), "Pred_Probab_N.csv"))