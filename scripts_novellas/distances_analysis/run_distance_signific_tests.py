system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')



from preprocessing.distances import iterate_inter_tests, iterate_intra_tests, IterateDistanceCalc, rd_iterate_intra_tests, GroupDistances, InterGroupDistances, random_iterate_rel_ttest, iterate_rel_ttest
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





filename = "red-to-2500mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
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

filename_base = os.path.splitext(filename)[0]


metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

label_list = ["R", "M", "E", "N", "0E", "XE"]
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"
name_cat = "Nachname"

metric = "cosine"
n=10

outfile_csv_path = str(n) + "_" + metric + "_distances_" + filename
outfile_dict_txt_path = str(n) + "_" + metric + "_distances_" + filename_base + ".txt"
outfile_results_df_path = os.path.join(local_temp_directory(system), outfile_csv_path)
outfile_dict_path = os.path.join(local_temp_directory(system), outfile_dict_txt_path)

dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)
dtm_obj = dtm_obj.add_metadata([genre_cat, year_cat, name_cat])
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge", "de", "di"])

dtm_obj.data_matrix_df = dtm_obj.data_matrix_df.drop(["00475-00", "00349-00", "00490-00", "00580-00"]) # remove Dubletten and texts that are intentionally represented twice in the corpus

dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1790, end_year=1950, epoch_length=1,
                                          new_periods_column_name="periods")
dtm_obj = dtm_obj.eliminate([year_cat])

list_of_real_genres_dfs = []

label_list = ["N", "E", "M", "R", "0E", "XE"]
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[0]])
df_N = genre_dtm_obj.data_matrix_df
list_of_real_genres_dfs.append([label_list[0], df_N])
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[1]])
df_E = genre_dtm_obj.data_matrix_df
list_of_real_genres_dfs.append([label_list[1], df_E])
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[2]])
df_M = genre_dtm_obj.data_matrix_df
list_of_real_genres_dfs.append([label_list[2], df_M])
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[3]])
df_R = genre_dtm_obj.data_matrix_df
list_of_real_genres_dfs.append([label_list[3], df_R])
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[4]])
df_0E = genre_dtm_obj.data_matrix_df
list_of_real_genres_dfs.append([label_list[4], df_0E])

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[5]])
df_XE = genre_dtm_obj.data_matrix_df
list_of_real_genres_dfs.append([label_list[5], df_XE])

bool_select_one_author = True
print("Select one author: ", str(bool_select_one_author))

bool_select_one_per_period = False
print("select one per period: ", str(bool_select_one_per_period))

df = dtm_obj.data_matrix_df.sample(frac=1.0)
rd_sample1, rd_sample2 = train_test_split(df, train_size=0.5)
df_mid_length_rd = pd.concat([df_N, df_E, df_XE, df_0E]).sample(frac=1.0)
print(df_mid_length_rd)


print("sample sizes: one per period and one author:")
print("Sample size Novellen (sample one instance per author): ", len(sample_n_from_cat(sample_n_from_cat(df_N), cat_name="periods")))
print("Sample size Erzählungen (sample one instance per author): ", len(sample_n_from_cat(sample_n_from_cat(df_E), cat_name="periods")))
print("Sample size Märchen (sample one instance per author): ", len(sample_n_from_cat(sample_n_from_cat(df_M), cat_name="periods")))
print("Sample size Romane (sample one instance per author): ", len(sample_n_from_cat(sample_n_from_cat(df_R), cat_name="periods")))
print("Sample size 0E (sample one instance per author): ", len(sample_n_from_cat(sample_n_from_cat(df_0E), cat_name="periods")))
print("Sample size XE (sample one instance per author): ", len(sample_n_from_cat(sample_n_from_cat(df_XE), cat_name="periods")))
print("len rd_sample1 from 150", len((sample_n_from_cat(rd_sample1.sample(150), cat_name="periods"))))
print("len rd_sample2", len((sample_n_from_cat(rd_sample2, cat_name="periods"))))
print("len mid length prose rd_sample from 150", len((sample_n_from_cat(df_mid_length_rd.sample(150), cat_name="periods"))))

print("sample sizes: one per author (no regulation of time signal):")
print("Sample size Novellen (sample one instance per author): ", len(sample_n_from_cat(df_N)))
print("Sample size Erzählungen (sample one instance per author): ", len(sample_n_from_cat(df_E)))
print("Sample size Märchen (sample one instance per author): ", len(sample_n_from_cat(df_M)))
print("Sample size Romane (sample one instance per author): ", len(sample_n_from_cat(df_R)))
print("Sample size 0E (sample one instance per author): ", len(sample_n_from_cat(df_0E)))
print("Sample size XE (sample one instance per author): ", len(sample_n_from_cat(df_XE)))
print("len rd_sample1 from sample n=150", len(sample_n_from_cat(rd_sample1.sample(150))))
print("len rd_sample2", len(sample_n_from_cat(rd_sample2)))
print("len mid length prose rd_sample from sample n=150", len(sample_n_from_cat(df_mid_length_rd.sample(150))))


n = 1
N_dists = IterateDistanceCalc(n, input_df_1=df_N, input_df_2=None, metric=metric, select_one_author=True).distances
print("len N dists:", len(N_dists))
E_dists = IterateDistanceCalc(n, input_df_1=df_E, metric=metric, select_one_author=True).distances
print("len E dists: ", len(E_dists))
R_dists = IterateDistanceCalc(n, input_df_1= df_R, metric= metric, select_one_author=True).distances
print("len R dists:", len(R_dists))
M_dists = IterateDistanceCalc(n, input_df_1=df_M, metric=metric, select_one_author=True).distances
print("len M dists: ", len(M_dists))
rd1_dists = IterateDistanceCalc(n, input_df_1=rd_sample1, metric=metric, select_one_author=True).distances
rd2_dists = IterateDistanceCalc(n, input_df_1=rd_sample2, metric=metric, select_one_author=True).distances
print("len rd_dists: ", len(rd1_dists))
N_E_dists = IterateDistanceCalc(n, df_N, df_E, metric, select_one_author=True).distances
print("len N-E-inter dists: ", len(N_E_dists))
E_N_dists = IterateDistanceCalc(n, df_E, df_N, metric, select_one_author=True).distances
print("len E-N-inter dists: ", len(N_E_dists))
R_M_dists = IterateDistanceCalc(n, df_R, df_M, metric, select_one_author=True).distances
print("len R-M-inter dists: ", len(R_M_dists))
rd_inter_dists = IterateDistanceCalc(n, rd_sample1, rd_sample2, metric, select_one_author=True).distances
print("len rd_inter_dists", len(rd_inter_dists))

n=10
p_to_df1, F_to_df1, p_to_df2, F_to_df2 = rd_iterate_intra_tests(n, df,
                                                                metric=metric,
                                                                alternative="less",
                                                                test_function=stats.mannwhitneyu,
                                                                select_one_author=bool_select_one_author,
                                                                select_one_per_period=bool_select_one_per_period,
                                                                sample_size_df=50)
print("average over n samples: rd_all_prose1 vs. rd_all_prose2 D(intra), sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = rd_iterate_intra_tests(n, df_mid_length_rd,
                                                                metric=metric,
                                                                alternative="less",
                                                                test_function=stats.mannwhitneyu,
                                                                select_one_author=bool_select_one_author,
                                                                select_one_per_period=bool_select_one_per_period,
                                                                sample_size_df=50)
print("average over n samples: rd_mid-length_prose1 vs. rd_mid-length_prose2 D(intra), sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())

for genre_label, df in list_of_real_genres_dfs:
    p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_intra_tests(n, df, df_mid_length_rd,
                                            metric=metric, alternative="less",
                                            test_function=stats.mannwhitneyu,
                                            select_one_author=bool_select_one_author,
                                            select_one_per_period=bool_select_one_per_period,
                                            smaller_sample_size=False, sample_size_df_1=None, sample_size_df_2=None)
    print("average over n samples: " + genre_label  + " vs. rd sample sign. MannWhitneyU Test, test-statistic, p-value: ")
    print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())


p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_inter_tests(n, df_N, df_E,
                                    metric=metric, alternative="greater",
                                    test_function=stats.mannwhitneyu,
                                    select_one_author=bool_select_one_author,
                                    select_one_per_period=bool_select_one_per_period,
                                    smaller_sample_size=False, sample_size_df_1=None, sample_size_df_2=None)
print("average over n samples: N_E vs N sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())
print(" average over n samples: N_E vs E sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df2).mean(), np.array(p_to_df2).mean())



p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_inter_tests(n, df_R, df_M,
                        metric=metric, alternative="greater", test_function=stats.mannwhitneyu,
                        select_one_per_period=bool_select_one_per_period,
                        select_one_author=bool_select_one_author,
                        smaller_sample_size=False, sample_size_df_1=None, sample_size_df_2=None)
print("average over n samples: R_M vs R sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())
print(" average over n samples: R_M vs M sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df2).mean(), np.array(p_to_df2).mean())

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_inter_tests(n, rd_sample1.sample(120), rd_sample2.sample(120), metric=metric, alternative="greater", test_function=stats.mannwhitneyu, select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period)
print("average over n samples: rd1_all vs rd2_all D(inter) sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())
print(" average over n samples: rd vs rd2 sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df2).mean(), np.array(p_to_df2).mean())





p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_intra_tests(n, df_E, df_mid_length_rd, metric=metric, alternative="less",
                                                             test_function=stats.mannwhitneyu, select_one_author=bool_select_one_author,
                                                             select_one_per_period=bool_select_one_per_period,
                                                             smaller_sample_size=False)
print("average over n samples: E vs. rd sample sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_intra_tests(n, df_M, rd_sample1.sample(50), metric=metric, alternative="less", test_function=stats.mannwhitneyu, select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period)
print("average over n samples: M vs. rd sample sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_intra_tests(n, df_R, rd_sample1.sample(50), metric=metric, alternative="less", test_function=stats.mannwhitneyu, select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period)
print("average over n samples: R vs. rd sample sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())