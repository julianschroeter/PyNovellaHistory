system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')



from metrics.distances import iterate_inter_tests, iterate_intra_tests, IterateDistanceCalc, rd_iterate_intra_tests, GroupDistances, InterGroupDistances, random_iterate_rel_ttest, iterate_rel_ttest
from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory, local_temp_directory
from preprocessing.metadata_transformation import full_genre_labels, years_to_periods
from preprocessing.sampling import split_to2samples, sample_n_from_cat

import os
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.presetting import local_temp_directory
import pandas as pd
import copy
from itertools import combinations
import json
from sklearn.model_selection import train_test_split
from scipy import stats






filename = "red-to-100mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename= "RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-2500mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-1000mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-1000mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-500mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-100mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_no-stopwords_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-2500mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"

filename_base = os.path.splitext(filename)[0]


metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

genre_label_list = ["E", "N", "0E", "XE"]
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"
name_cat = "Nachname"
class_cat = "Kanon_Status"

metric = "cosine"
n=10

outfile_csv_path = str(n) + "_" + metric + "_distances_" + filename
outfile_dict_txt_path = str(n) + "_" + metric + "_distances_" + filename_base + ".txt"
outfile_results_df_path = os.path.join(local_temp_directory(system), outfile_csv_path)
outfile_dict_path = os.path.join(local_temp_directory(system), outfile_dict_txt_path)

dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)
dtm_obj = dtm_obj.add_metadata([genre_cat, year_cat, name_cat, class_cat])
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge", "de", "di"])

dtm_obj.data_matrix_df = dtm_obj.data_matrix_df.drop(["00475-00", "00349-00", "00490-00", "00580-00"]) # remove Dubletten and texts that are intentionally represented twice in the corpus

dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1790, end_year=1950, epoch_length=1,
                                          new_periods_column_name="periods")
dtm_obj = dtm_obj.eliminate([year_cat])

list_of_real_genres_dfs = ["Kanon-Novellen", "sonstige MLP"]


dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=genre_label_list)
dtm_obj = dtm_obj.eliminate([genre_cat])


df = dtm_obj.data_matrix_df

print(df)

replace_dict = {class_cat: {3: "Kanon-Novelle", 2: "Kanon-Novelle", 1: "sonstige MLP", 0: "sonstige MLP"}}

dtm_obj.data_matrix_df = full_genre_labels(dtm_obj.data_matrix_df, replace_dict=replace_dict)


label_list = ["Kanon-Novelle", "sonstige MLP"]

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=class_cat, label_list=[label_list[0]])
df_NovSch = genre_dtm_obj.data_matrix_df

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=class_cat, label_list=[label_list[1]])
df_other = genre_dtm_obj.data_matrix_df



bool_select_one_author = True
print("Select one author: ", str(bool_select_one_author))
bool_select_one_per_period = False
print("Select one period: ", str(bool_select_one_per_period))

#df = dtm_obj.data_matrix_df.sample(frac=1.0)
rd_sample1, rd_sample2 = train_test_split(df, train_size=0.5)




n=10
p_to_df1, F_to_df1, p_to_df2, F_to_df2 = rd_iterate_intra_tests(n, df,
                                                                metric=metric,
                                                                alternative="less",
                                                                test_function=stats.mannwhitneyu,
                                                                select_one_author=bool_select_one_author,
                                                                select_one_per_period=bool_select_one_per_period,
                                                                sample_size_df=50)
print("average over n samples: Kanon vs. non-Kanon, sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())



p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_intra_tests(n, df_NovSch, df_other,
                                        metric=metric, alternative="greater",
                                        test_function=stats.mannwhitneyu,
                                        select_one_author=bool_select_one_author,
                                        select_one_per_period=bool_select_one_per_period,
                                        smaller_sample_size=False, sample_size_df_1=None, sample_size_df_2=None)

print("average over n samples: Kanon3 vs Rest sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())
print(" average over n samples: rest vs Kanon3 sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df2).mean(), np.array(p_to_df2).mean())
