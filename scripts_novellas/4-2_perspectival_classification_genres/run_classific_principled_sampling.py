system = "my_xps" # "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os
import numpy as np
from collections import defaultdict

from preprocessing.presetting import global_corpus_representation_directory, global_corpus_directory, language_model_path, vocab_lists_dicts_directory, word_translate_table_to_dict, global_corpus_raw_dtm_directory, local_temp_directory
from preprocessing.corpus import DTM
from preprocessing.sampling import principled_sampling
from classification.perspectivalmodeling import split_features_labels
from clustering.my_plots import plot_prototype_concepts
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd

from metrics.scores import c_at_1

metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

label_list = ["N", "E"]

name_cat = "Nachname"
periods_cat = "Jahr_ED"
genre_cat = "Gattungslabel_ED_normalisiert"

probabs_list, all_coef_features = [], []
probabs_dict = {}
acc_scores = []
optima_x, optima_y = [],[]

n= 100
for filename in os.listdir(global_corpus_raw_dtm_directory(system)):
    if filename == "raw_dtm_l1_lemmatized_use_idf_False2500mfw.csv": # alternative: "no-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
        filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)


        for i in range(n):
            dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)
            dtm_obj.data_matrix_df = dtm_obj.data_matrix_df.sample(frac=0.9, axis="columns")

            dtm_obj = dtm_obj.add_metadata([genre_cat, name_cat, periods_cat])
            dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=label_list)
            dtm_obj = dtm_obj.eliminate(["roman", "märchen", "fle", "be", "te", "ge"])

            df = dtm_obj.data_matrix_df
            df_0 = df[df[genre_cat] == "M"]
            df_1 = df[df[genre_cat] == "N"]

            train_set, test_set = principled_sampling(df_1, df_0)

            year_labels = dtm_obj.data_matrix_df["Jahr_ED"].to_list()
            dtm_obj = dtm_obj.eliminate(["Jahr_ED"])

            indexes = df.index.tolist()
            features = df.columns.tolist()

            X_train, Y_train_orig = split_features_labels(train_set)
            X_test, Y_test_orig = split_features_labels(test_set)
            print(X_train)
            print(Y_train_orig)

            subs_dict = {"R": 1, "M": 0}
            Y_train = list(map(subs_dict.get, Y_train_orig, Y_train_orig))
            Y_test = list(map(subs_dict.get, Y_test_orig, Y_test_orig))

            lr_model = LogisticRegressionCV(cv=3, solver='liblinear', multi_class="auto")
            lr_model.fit(X_train, Y_train)
            test_predictions = lr_model.predict(X_test)

            coef = lr_model.coef_.tolist()
            coef = [item for sublist in coef for item in sublist]


            coef_features = list(zip(features, coef))
            all_coef_features.extend(coef_features)
            #print("coef with features: ", sorted(coef_features, key=lambda x: x[0]))

            acc_scores.append(lr_model.score(X_test, Y_test))

            print(classification_report(Y_test, test_predictions))
            predict_probs_inv = [x for sublist in lr_model.predict_proba(X_test)[:,0:1] for x in  sublist]
            predict_probs = [x for sublist in lr_model.predict_proba(X_test)[:, 1:2] for x in sublist]

            new_probabs_list = list(zip(test_set.index.values, predict_probs))
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

thresholds_scores_plot = thresholds_scores

d = defaultdict(list)

for k, v in probabs_list:
    d[k].append(v)
probabs_list = sorted(d.items())

d = defaultdict(list)
for k, v in all_coef_features:
    d[k].append(v)
all_coef_features = sorted(d.items())


dictionary = defaultdict(list)
for entry in all_coef_features:
    values = np.array(entry[1])
    mean = np.mean(values)
    std = np.std(values)
    dictionary[entry[0]] = [mean, std]

df = pd.DataFrame.from_dict(dictionary, orient="index", columns=["mean", "st_dev"])
df = df.sort_values(by="mean")
print(df)
df.to_csv(path_or_buf=os.path.join(local_temp_directory(system), "av_features_coefs_n100_N-M.csv"))

dictionary = defaultdict(list)
for entry in probabs_list:
    values = np.array(entry[1])
    mean = np.mean(values)
    std = np.std(values)
    dictionary[entry[0]] = [mean, std]

df = pd.DataFrame.from_dict(dictionary, orient="index", columns=["mean", "st_dev"])
df = df.sort_values(by="mean")

obj = DTM(data_matrix_df=df, metadata_csv_filepath=metadata_path)
obj = obj.add_metadata([genre_cat, name_cat, "Titel"])
df = obj.data_matrix_df
print(df)
means = df["mean"].values.tolist()
predict_probs_inv = [1 - value for value in means]
predict_probs = means
labels = df[genre_cat].values.tolist()

subs_dict = {"N": "red", "M": "orange"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

subs_dict = {"red": 1, "orange": 0}
genre_bin = list(map(subs_dict.get, labels, labels))

#threshold_ranges = np.arange(0.00, 1, 0.01)
#thresholds_scores = []
#for threshold in threshold_ranges:
#    scores = (threshold, c_at_1(genre_bin, predict_probs, threshold))
#    thresholds_scores.append(scores)
scores = [scores[1] for scores in thresholds_scores]
final_optimum = max(thresholds_scores, key=lambda scores: scores[1])

optimum_x = np.mean(np.array(optima_x))
optimum_Y = np.mean(np.array(optima_y))

str1 = "optimum x: " + str(optimum_x)
str2 = "optimum y: " + str(optimum_Y)

print(thresholds_scores_plot)

plt.vlines(optimum_x, ymin=0, ymax=optimum_Y, colors="blue")
plt.hlines(optimum_Y, xmin=0, xmax=1, colors="pink")
plt.plot([e[0] for e in thresholds_scores_plot], [e[1] for e in thresholds_scores_plot])
plt.ylabel("c@1-accuracy scores")
plt.xlabel("Breite des Bereichs der Unentscheidbarkeit")
plt.title("Gridsuche: c@1-Sore und Bereich der Unentscheidbarkeit")
plt.savefig("/home/julian/git/PyNovellaHistory/figures/gridsuche_c-at-one_R-M.svg")
plt.show()


outfile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_N-M_n100.txt")
with open(outfile_output, "w") as file:
    file.write(str1 +"\n")
    file.write(str2)
    file.close()

#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]

df.to_csv(path_or_buf=os.path.join(local_temp_directory(system), "av_pred_probabs_N-M_n100.csv"))

#plot_prototype_concepts(predict_probs_inv, genre_c_labels, threshold=optimum_x, annotation=annotation)

#romeo_prototyp = df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]
plot_prototype_concepts(predict_probs, genre_c_labels, threshold=optimum_x)
