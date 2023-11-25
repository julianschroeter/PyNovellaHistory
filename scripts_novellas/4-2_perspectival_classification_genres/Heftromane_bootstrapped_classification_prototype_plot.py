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

metadata_path = os.path.join(local_temp_directory(system), "Heftromane", "Metadatendatei_Heftromane.tsv")

label_list = ["N", "E"]

name_cat = "Nachname"
periods_cat = "date"
genre_cat = "genre"

probabs_list, all_coef_features = [], []
probabs_dict = {}
acc_scores = []
optima_x, optima_y = [],[]

n= 1000
for filename in os.listdir(os.path.join(local_temp_directory(system), "Heftromane", "misc")):
    if filename == "dtm_8000mfw.tsv": # alternative: "no-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
        filepath = os.path.join(os.path.join(local_temp_directory(system), "Heftromane", "misc"), filename)

        for i in range(n):
            df = pd.read_csv(filepath, sep=",", index_col=0).T
            print(df.index)
            metadata_df = pd.read_csv(metadata_path, sep="\t", index_col=0)
            df["genre"] = df.apply(lambda x: metadata_df.loc[x.name, "genre"], axis=1)
            df["texttype"] = df["genre"].apply(lambda x: "spannung" if x in ("horror", "krimi", "krieg", "western", "scifi", "abenteuer")
                else ("liebe" if x in ("liebe", "arzt", "heimat", "adel", "erotik") else "other"))
            df["date"] = df.apply(lambda x: metadata_df.loc[x.name, "date"], axis=1)
            df["Nachname"] = df.apply(lambda x: metadata_df.loc[x.name, "author"], axis=1)
            df = df.dropna(subset=["date"])

            print(df)

            start_df = df

            df_spannung = df[df.isin({"texttype": ["spannung"]}).any(axis=1)]
            df_spannung["genre"] = df["texttype"].apply(lambda x: x)
            df_spannung.drop(columns=["texttype"], inplace=True)
            df_liebe = df[df.isin({"texttype": ["liebe"]}).any(axis=1)]
            df_liebe["genre"] = df["texttype"].apply(lambda x: x)
            df_liebe.drop(columns=["texttype"], inplace=True)

            train_set, test_set = principled_sampling(df_spannung, df_liebe, select_one_per_period=False, select_one_per_author=False)

            year_labels = df["date"].to_list()
            #df = df.drop(columns=["date"])

            indexes = df.index.tolist()
            features = df.columns.tolist()

            X_train, Y_train_orig = split_features_labels(train_set)
            X_test, Y_test_orig = split_features_labels(test_set)

            subs_dict = {"spannung": 1, "liebe": 0}
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
df.to_csv(path_or_buf=os.path.join(local_temp_directory(system), "av_features_coefs_n100_Heftromane.csv"))

dictionary = defaultdict(list)
for entry in probabs_list:
    values = np.array(entry[1])
    mean = np.mean(values)
    std = np.std(values)
    dictionary[entry[0]] = [mean, std]

df = pd.DataFrame.from_dict(dictionary, orient="index", columns=["mean", "st_dev"])
df = df.sort_values(by="mean")

df["genre"] = df.apply(lambda x: start_df.loc[x.name, "texttype"], axis=1)
df["date"] = df.apply(lambda x: metadata_df.loc[x.name, "date"], axis=1)
df["Nachname"] = df.apply(lambda x: metadata_df.loc[x.name, "author"], axis=1)

print(df)
means = df["mean"].values.tolist()
predict_probs_inv = [1 - value for value in means]
predict_probs = means
labels = df[genre_cat].values.tolist()
print(labels)

subs_dict = {"spannung": "red", "liebe": "green"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

subs_dict = {"red": 1, "green": 0}
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
plt.xlabel("Boundaries of Undecidability")
plt.title("Accuracy score at optimal boundary of undecidability")
plt.show()


outfile_output = os.path.join(local_temp_directory(system), "output_av_pred_probabs_Heftromane_n1000.txt")
with open(outfile_output, "w") as file:
    file.write(str1 +"\n")
    file.write(str2)
    file.close()

#romeo_prototyp = 1 - df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]

df.to_csv(path_or_buf=os.path.join(local_temp_directory(system), "av_pred_probabs_Heftromane_n1000.csv"))

#plot_prototype_concepts(predict_probs_inv, genre_c_labels, threshold=optimum_x, annotation=annotation)

#romeo_prototyp = df.loc["00306-00", "mean"]
#annotation = ["Romeo und Julia auf dem Dorfe", romeo_prototyp]
plot_prototype_concepts(predict_probs, genre_c_labels, threshold=optimum_x)
