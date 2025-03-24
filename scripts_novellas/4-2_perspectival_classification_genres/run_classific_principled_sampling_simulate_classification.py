system = "my_xps" # "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import numpy as np
from collections import defaultdict
from clustering.my_plots import plot_prototype_concepts, rd_vectors_around_center
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pandas as pd
from metrics.scores import c_at_1
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

lang = "de"
n = 300

flip_y = 0.1
class_sep = 2

fig, axes = plt.subplots(5, 3, figsize=(15,25))

probabs_list, all_coef_features = [], []
probabs_dict = {}
acc_scores = []
optima_x, optima_y = [], []
improvements = []
lin_scores, nonlin_scores = [], []

for i in range(n):
    if i < 100:
        n_informative = 2
        n_redundant = 0
        n_clusters_per_class = 1
        title = "2 informative Features, 1 Cluster"

        k = 0

    elif i < 200:
        n_informative = 2
        n_redundant = 0
        n_clusters_per_class = 2
        title = "2 informative Features, 2 Cluster"
        k = 1


    else:
        n_informative = 3
        n_redundant = 0
        n_clusters_per_class = 3
        title = "3 informative Features, 3 Cluster"
        k = 2

    if  i in [100,200]:
        probabs_list, all_coef_features = [], []
        probabs_dict = {}
        acc_scores = []
        optima_x, optima_y = [], []
        improvements = []
        lin_scores, nonlin_scores = [], []

    X, Y = make_classification(n_samples=500, flip_y=flip_y, class_sep=class_sep,
                               n_features=5, n_redundant=n_redundant, n_informative=n_informative, n_clusters_per_class=n_clusters_per_class)


    pca = PCA(n_components=0.95)
    vecs = pca.fit_transform(X,Y)
    if i in [0,100,200]:
        axes[0,k].scatter(vecs[:, 0], vecs[:, 1], marker="o", c=Y, s=25, edgecolor="k")
        axes[0,k].set_title(title, fontsize="medium")
        axes[0,0].set_ylabel("PCA")




    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    lr_model = LogisticRegressionCV(cv=3, solver='liblinear', multi_class="auto")
    lr_model = LogisticRegression()

    lr_model.fit(X_train, Y_train)
    test_predictions = lr_model.predict(X_test)

    coef = lr_model.coef_.tolist()
    coef = [item for sublist in coef for item in sublist]


    acc_scores.append(lr_model.score(X_test, Y_test))

    print(classification_report(Y_test, test_predictions))
    predict_probs_inv = [x for sublist in lr_model.predict_proba(X_test)[:,0:1] for x in  sublist]
    predict_probs = [x for sublist in lr_model.predict_proba(X_test)[:, 1:2] for x in sublist]

    new_probabs_list = list(zip(Y_test.tolist(), predict_probs))
    probabs_list.extend(new_probabs_list)

    threshold_ranges = np.arange(0.00, 1, 0.01)
    thresholds_scores = []
    for threshold in threshold_ranges:
        scores = (threshold, c_at_1(Y_test, predict_probs, threshold))

        thresholds_scores.append(scores)
    scores = [scores[1] for scores in thresholds_scores]

    optimum = max(thresholds_scores, key=lambda scores: scores[1])
    optima_x.append(optimum[0])
    optima_y.append(optimum[1])

    print("mean of accuracy scores: ", np.mean(acc_scores))


    labels = Y_test.tolist()

    subs_dict = {"1": "red", "0": "orange"}
    genre_c_labels = list(map(subs_dict.get, labels, labels))


    scores = [scores[1] for scores in thresholds_scores]
    final_optimum = max(thresholds_scores, key=lambda scores: scores[1])

    optimum_x = np.mean(np.array(optima_x))
    optimum_Y = np.mean(np.array(optima_y))

    str1 = "optimum x: " + str(optimum_x)
    str2 = "optimum y: " + str(optimum_Y)

    print(thresholds_scores)
    improvement = optimum_Y / np.mean(acc_scores)
    improvements.append(improvement)
    print("Improvement: ", title, ": ", improvement)

    lin_model = LinearSVC()
    nonlin_model = SVC()


    # train a linear SVM :
    lin_model.fit(X_train, Y_train)
    Y_pred_linear = lin_model.predict(X_test)
    print("Linear SVM Classification Report:")
    print(classification_report(Y_test, Y_pred_linear))
    print("Accuracy: ", accuracy_score(Y_test, Y_pred_linear))
    lin_scores.append(accuracy_score(Y_test, Y_pred_linear))

    # Train a non-linear SVM with RBF kernel and standard parameter settings
    nonlinear_svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))
    nonlinear_svm.fit(X_train, Y_train)
    Y_pred = nonlinear_svm.predict(X_test)
    print("Non-linear SVM Classification Report:")
    print(classification_report(Y_test, Y_pred))
    print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.2f}")
    nonlin_scores.append(accuracy_score(Y_test, Y_pred))

    lin_mean = np.mean(np.asarray(lin_scores))
    print("Mean accuracy for linear SVM: ", lin_mean)

    nonlin_mean = np.mean(np.asarray(nonlin_scores))
    print("Mean accuracy for non-linear SVM: ", nonlin_mean)



    if i in [0,100,200]:
        axes[1,k].plot([e[0] for e in thresholds_scores], [e[1] for e in thresholds_scores])
        axes[1,k].vlines(optimum_x, ymin=0, ymax=optimum_Y, colors="blue")
        axes[1,k].hlines(optimum_Y, xmin=0, xmax=1, colors="pink")
        axes[1,k].set_ylim(0,1)
        axes[1,0].set_ylabel("c@1")
        axes[1,1].set_xlabel("Enthaltungsbereich")
        axes[1,k].set_title("accuracy: " + str(np.mean(acc_scores))[:4] + " – c@1: " + str(optimum_Y)[:4])

    threshold = optimum_x
    x_results, y_results = rd_vectors_around_center(predict_probs)
    lower_threshold_level = 0.5 - (threshold / 2)
    upper_threshold_level = 0.5 + (threshold / 2)

    if i in [0,100,200]:
        axes[2,k].scatter(x_results, y_results, c=labels)
        axes[2,k].add_patch(plt.Circle((0, 0), 1, fill=False))
        axes[2,k].add_patch(plt.Circle((0, 0), lower_threshold_level, fill=False))
        axes[2,k].add_patch(plt.Circle((0, 0), upper_threshold_level, fill=False))

        rd_angle = np.random.uniform(-np.pi, np.pi, 1)
        if lang == "de":
            axes[2,1].set_title("Prototypenkonzept für Genre-Paare")
            axes[2,k].set_xlabel(str("Grenzbereich der Unentscheidbarkeit: " + str(lower_threshold_level) + " – " + str(
                upper_threshold_level)))
            axes[2,0].set_ylabel("Nähe zum Prototypenzentrum (inv. Vorhersagewahrsch.)")

        if lang == "en":
            plt.title("Prototype Concept for Genre Pairs")
            plt.xlabel(str("Boundary of undecidabilty: " + str("%.2f" % round(lower_threshold_level,2)) + " – " + str("%.2f" % round(upper_threshold_level, 2))))
            plt.ylabel("closeness to center (inv. pred. prob.)")

        axes[2,k].set_xlim(-1, 1)
        axes[2,k].set_ylim(-1, 1)

        print(optima_x)
    if i in [99, 199, 299]:
        axes[3,k].boxplot(optima_x, vert=True)
        axes[3, k].set_ylim(0,1)
        axes[3,0].set_ylabel("Boxplot: Enthaltungsbereiche")
        axes[3, 1].set_xlabel("Enthaltungsbereich")
        axes[4, k].boxplot(improvements, vert=True)
        axes[4,k].set_ylim(1.00,1.06)
        axes[4,0].set_ylabel("Boxplot: Verbesserung")
        axes[4, k].set_xlabel("LR accuracy: " + str(np.mean(acc_scores))[:4] + " – c@1: " + str(optimum_Y)[:4] +"\n" +
        "SVC lin acc: " + str(lin_mean)[:4] + " / Non-Lin SVC acc: " + str(nonlin_mean)[:4] + "\n" +
                              "Nonlin Impr.Rate: " + str(nonlin_mean/lin_mean)[:4])


fig.tight_layout()
fig.savefig("/home/julian/Documents/CLS_temp/figures/5features_gridsuche_c-at-one_simulate_class_sep-" + str(class_sep) + "flip_y-" + str(flip_y) + ".svg")
plt.show()

