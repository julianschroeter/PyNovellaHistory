system = "my_xps" # "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os
import numpy as np
from numpy import unique
from numpy import where


from preprocessing.presetting import global_corpus_representation_directory, global_corpus_directory, language_model_path, vocab_lists_dicts_directory, word_translate_table_to_dict, global_corpus_raw_dtm_directory, local_temp_directory
from preprocessing.corpus import DTM
from classification.perspectivalmodeling import split_features_labels
from clustering.my_plots import plot_prototype_concepts
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

from metrics.scores import c_at_1

metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

label_list = ["N", "E"]


for filename in os.listdir(global_corpus_raw_dtm_directory(system)):
    if filename ==  "no-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv": # raw_dtm_l1_lemmatized_use_idf_False2500mfw.csv": # "no-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv":
        filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)
        dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)

        dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED"])
        dtm_obj = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=label_list)
        dtm_obj = dtm_obj.eliminate(["roman", "märchen", "novelle", "erzählung","fle", "be", "te", "ge"])
        year_labels = dtm_obj.data_matrix_df["Jahr_ED"].to_list()
        dtm_obj = dtm_obj.eliminate(["Jahr_ED"])

        df = dtm_obj.data_matrix_df
        indexes = df.index.tolist()
        features = df.columns.tolist()

        X, Y_orig = split_features_labels(df)

        subs_dict = {"N": 1, "E": 0}
        Y = list(map(subs_dict.get, Y_orig, Y_orig))

        subs_dict = {0: "green", 1: "red"}
        Y_color = list(map(subs_dict.get, Y, Y))

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8, random_state=42)

        kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_train)
                        for k in range(1, 10)]
        silhouette_scores = [silhouette_score(X_train, model.labels_)
                             for model in kmeans_per_k[1:]]

        print(silhouette_scores)

        plt.figure(figsize=(8, 3))
        plt.plot(range(2, 10), silhouette_scores, "bo-")
        plt.xlabel("$k$", fontsize=14)
        plt.ylabel("Silhouette score", fontsize=14)
        plt.title("Optimale Anzahl k Cluster")
        plt.xlabel("Anzahl k der Cluster")
        plt.axis([1.8, 8.5, 0.0, 0.5])
        plt.tight_layout()
        plt.savefig(os.path.join(local_temp_directory(system), "figures", "silhouette_score_vs_k_plot.svg"))
        plt.show()

        kmeans = KMeans(n_clusters=2)

        # assign each data point to a cluster
        result = kmeans.fit_predict(X)
        print(kmeans.cluster_centers_)

        print("clustering accuracy: ", accuracy_score(Y, result))

        # get all of the unique clusters
        clusters = unique(result)

        fig, axes = plt.subplots(1,2)
        pca = PCA(n_components=0.95)
        vecs = pca.fit_transform(X, Y)
        # plot the clusters
        for cluster in clusters:
            # get data points that fall in this cluster
            index = where(result == cluster)
            # make the plot
            axes[1].scatter(vecs[index, 0], vecs[index, 1])
        axes[1].set_title("Clustering mi K-Means")
        axes[1].set_xlabel("Unüberwachte Gruppierung")
        legend_dict = {"Novelle": "red", "Erzählung": "green"}
        mpatches_list = []
        for key, value in legend_dict.items():
            patch = mpatches.Patch(color=value, label=key)
            mpatches_list.append(patch)
        axes[0].legend(handles=mpatches_list)


        axes[0].scatter(vecs[:,0], vecs[:,1], c="None", edgecolor=Y_color)
        axes[0].set_title("PCA")
        axes[0].set_xlabel("Mit Labels der Erstdrucke")
        fig.tight_layout()
        fig.savefig(os.path.join(local_temp_directory(system), "figures", "K-Means-Clustering-ana-PCA_N-E.svg"))
        plt.show()

        rnd_model = RandomForestClassifier()
        lr_model = LogisticRegressionCV()
        rnd_model.fit(X_train, Y_train)
        lr_model.fit(X_train, Y_train)
        test_predictions = rnd_model.predict(X_test)

        feat_importance = rnd_model.feature_importances_.tolist()


        importance_features = list(zip(feat_importance, features))
        print("importances with feature names: ", sorted(importance_features, key=lambda x: x[0]))

        print("classification report with regular train/test sets for " + str(filename) + ": ")
        print(rnd_model.score(X, Y))
        print(accuracy_score(Y_test, test_predictions))
        print(classification_report(Y_test, test_predictions))

        print("Log Reg Baseline Model:")
        lr_test_predictions = lr_model.predict(X_test)
        print(classification_report(Y_test, lr_test_predictions))

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

        from pprint import pprint  # Look at parameters used by our current forest

        print('Parameters currently in use:\n')
        pprint(rnd_model.get_params())

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

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune

        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rnd_model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)  # Fit the random search model
        rf_random.fit(X_train, Y_train)
        test_predictions = rf_random.best_estimator_.predict(X_test)

        print(rf_random.best_params_)
        print(accuracy_score(Y_test, test_predictions))

