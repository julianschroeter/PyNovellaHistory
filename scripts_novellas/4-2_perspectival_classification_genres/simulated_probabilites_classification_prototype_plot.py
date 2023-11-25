system = "my_xps" # "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os
import numpy as np
from collections import defaultdict
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_directory, language_model_path, vocab_lists_dicts_directory, word_translate_table_to_dict, global_corpus_raw_dtm_directory, local_temp_directory
from clustering.my_plots import plot_prototype_concepts
from sklearn import model_selection
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

lower = np.random.randint(low=0, high=40, size=300)
upper = np.random.randint(low=60,high=100,size= 300)
between = np.random.randint(low=40, high=60,size= 400)

X = np.concatenate((lower, between, upper), axis=0)
X = np.divide(X, 100)
print(X)

labels = []
start = 0
for i in range(0,1000):
    if start < 300:
        if (i/10).is_integer():
            e = 0
        else:
            e = 0
    elif 400 <= start < 700:
        e = np.random.randint(0,2)
    else:
        if (i/10).is_integer():
            e = 1
        else:
            e = 1
    start += 1
    labels.append(e)
print(labels)
Y = labels

n=100
for i in range(n):
    X_train, X_test, Y_train,  Y_test = model_selection.train_test_split(X,Y, train_size=0.5)
    predict_probs = X_train
    labels = Y_train
    threshold_ranges = np.arange(0.00, 1, 0.01)
    thresholds_scores = []
    for threshold in threshold_ranges:
        scores = (threshold, c_at_1(labels, predict_probs, threshold))
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
#df.to_csv(path_or_buf=os.path.join(local_temp_directory(system), "av_features_coefs_n100_Heftromane.csv"))

dictionary = defaultdict(list)
for entry in probabs_list:
    values = np.array(entry[1])
    mean = np.mean(values)
    std = np.std(values)
    dictionary[entry[0]] = [mean, std]

df = pd.DataFrame.from_dict(dictionary, orient="index", columns=["mean", "st_dev"])
df = df.sort_values(by="mean")


print(df)
means = df["mean"].values.tolist()
predict_probs_inv = [1 - value for value in means]
#predict_probs = means

subs_dict = {1: "lightgreen", 0: "darkblue"}
genre_c_labels = list(map(subs_dict.get, labels, labels))

subs_dict = {"magenta": 1, "cyan": 0}
genre_bin = list(map(subs_dict.get, labels, labels))

scores = [scores[1] for scores in thresholds_scores]
final_optimum = max(thresholds_scores, key=lambda scores: scores[1])

optimum_x = np.mean(np.array(optima_x))
optimum_Y = np.mean(np.array(optima_y))

str1 = "optimum x: " + str(optimum_x)
str2 = "optimum y: " + str(optimum_Y)
print(str1, str2)
print(thresholds_scores_plot)

plot_prototype_concepts(predict_probs, genre_c_labels, threshold=optimum_x, lang="de",
                        legend_dict={"Genre C":"lightgreen","Genre D": "darkblue"},
                        filepath="/home/julian/git/PyNovellaHistory/figures/simulate_prototype_genres.svg")
