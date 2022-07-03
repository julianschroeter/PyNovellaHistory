from preprocessing.distances import InterGroupDistances, iterate_inter_tests

import pandas as pd
from scipy.spatial.distance import cdist

system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

from preprocessing.distances import IterateDistanceCalc, iterate_inter_tests
from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory
from preprocessing.metadata_transformation import years_to_periods
from preprocessing.sampling import sample_n_from_cat
import os
from scipy import stats
import numpy as np

metric = "cityblock"

df_1 = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], index=["T1", "T2", "T3"], columns=["w_a", "w_b", "w_c"])
print(df_1)

df_2 = pd.DataFrame([[1,2,3], [4,5,7], [7,8,10]], index=["T4", "T5", "T6"], columns=["w_a", "w_b", "w_c"])
print(df_2)


dist_obj = InterGroupDistances(df_1, df_2, metric=metric)
print(dist_obj.distances)
print(dist_obj.group_mean())

dist_obj = InterGroupDistances(df_2, df_1, metric=metric)
print(dist_obj.distances)
print(dist_obj.group_mean())


p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_inter_tests(1, df_1, df_2, metric=metric, alternative="greater", test_function=stats.mannwhitneyu, select_one_author=False)
print(p_to_df1, F_to_df1, p_to_df2, F_to_df2)

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_inter_tests(1, df_2, df_1, metric=metric, alternative="greater", test_function=stats.mannwhitneyu, select_one_author=False)
print(p_to_df1, F_to_df1, p_to_df2, F_to_df2)
# new from scratch:

dists = cdist(df_1, df_2, metric=metric)
print(dists)

dists = cdist(df_2, df_1, metric=metric)
print(dists)



filename = "scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

label_list = ["R", "M", "E", "N", "0E", "XE"]
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"

metric = "cosine"

dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)
dtm_obj = dtm_obj.add_metadata([genre_cat, year_cat, "Nachname"])
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge", "de", "di"])

dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1790, end_year=1950, epoch_length=1,
                                          new_periods_column_name="periods")
dtm_obj.eliminate([year_cat])

label_list = ["N", "E", "M", "R"]
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[0]])
#genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
genre_dtm_obj = genre_dtm_obj.eliminate(["periods"])
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_N = genre_dtm_obj.data_matrix_df
print("size of whole novellen sample: ", len(df_N))
print("Sample size Novellen (sample one instance per author): ", len(sample_n_from_cat(df_N)))

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[1]])
#genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
genre_dtm_obj = genre_dtm_obj.eliminate(["periods"])
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_E = genre_dtm_obj.data_matrix_df
print("Sample size Erzählungen (sample one instance per author): ", len(sample_n_from_cat(df_E)))


genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[2]])
#genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
genre_dtm_obj = genre_dtm_obj.eliminate(["periods"])
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_M = genre_dtm_obj.data_matrix_df
print("Sample size Märchen (sample one instance per author): ", len(sample_n_from_cat(df_M)))

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[3]])
#genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
genre_dtm_obj = genre_dtm_obj.eliminate(["periods"])
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_R = genre_dtm_obj.data_matrix_df
print("Sample size Romane (sample one instance per author): ", len(sample_n_from_cat(df_R)))

from sklearn.model_selection import train_test_split

rd_dtm_obj = dtm_obj.eliminate([year_cat, genre_cat, "periods"])
df = rd_dtm_obj.data_matrix_df
rd_sample1, rd_sample2 = train_test_split(df, train_size=0.5)

print("len rd_sample1", len((sample_n_from_cat(rd_sample1))))
print("len rd_sample2", len((sample_n_from_cat(rd_sample2))))


n = 10

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

print(np.array(N_E_dists).mean())
print(np.array(E_N_dists).mean())