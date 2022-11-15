system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os
from numpy import mean, std
import pandas as pd

from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory, local_temp_directory
from preprocessing.corpus_alt import DTM

from classification.custom_classification import resample_boostrapped_LR, resample_boostrapped_SVM

metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

label_list = ["N", "E"]

dic = {}

for filename in os.listdir(global_corpus_raw_dtm_directory(system)):
    filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)
    dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)

    dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert"])
    dtm_obj = dtm_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=label_list)
    dtm_obj = dtm_obj.eliminate(["novelle", "erzählung", "roman", "märchen", "ge", "te", "be"])
    df = dtm_obj.data_matrix_df


    mean_lr, std_lr = resample_boostrapped_LR(50, df, genre_labels=label_list)

    print("LR classification report for " + str(filename) + ": ")
    print(mean_lr, std_lr)

    mean_svm, std_svm = resample_boostrapped_SVM(50, df, genre_labels=label_list)

    print("SVM classification report for " + str(filename) + ": ")
    print(mean_svm, std_svm)

    norm, idf, scale, elimination, lemma, n_mfw, rfe = "", "", "", "", "", "", ""
    if "l1" in filename:
        norm = "l1"
    elif "l2" in filename:
        norm = "l2"
    if "use_idf_False" in filename:
        idf = False
    elif "use_idf_True" in filename:
        idf = True
    if "scaled" in filename:
        scale = "z"
    elif "scaled" not in filename:
        scale = False
    if "no-stopwords_no-names" in filename:
        elimination = "stopwords, names"
    elif "no-stopwords" in filename:
        elimination = "stopwords"
    elif "no-names" in filename:
        elimination = "names"
    else:
        elimination = False
    if "lemmatized" in filename:
        lemma = True
    else:
        lemma = False
    if "red-to-100mfw" in filename:
        n_mfw = 100
    elif "red-to-500mfw" in filename:
        n_mfw = 500
    elif "red-to-1000mfw" in filename:
        n_mfw = 1000
    elif "red-to-2500" in filename:
        n_mfw = 2500
    elif "RFE" in filename:
        n_mfw = "RFE_red"
    elif "red-to-" not in filename:
        n_mfw = 6000
    if "RFE" in filename:
        rfe = True
    else:
        rfe = False






    dic[filename] = [mean_lr,std_lr, mean_svm, std_svm, norm, idf, scale, elimination, lemma, n_mfw, rfe]


df = pd.DataFrame.from_dict(dic, orient="index", columns=["mean_lr", "std_lr", "mean_svm", "std_svm", "norm", "idf", "scaling", "eliminations", "lemmatization", "n_mfw", "RFE"])
df = df.round({"mean_lr":2, "std_lr":2, "mean_svm":2, "std_svm":2})
print(df)
df.to_csv(os.path.join(local_temp_directory(system), "bootstr-class-results_N-E.csv"))
