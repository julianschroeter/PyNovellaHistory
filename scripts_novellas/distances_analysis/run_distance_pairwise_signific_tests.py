system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')



from metrics.distances import IterateDistanceCalc, pairwise_self_counter_av_distances, GroupDistances, InterGroupDistances, random_iterate_rel_ttest, iterate_rel_ttest
from preprocessing.corpus_alt import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory, local_temp_directory
from preprocessing.metadata_transformation import full_genre_labels, years_to_periods
from preprocessing.sampling import split_to2samples, sample_n_from_cat

import os
from copy import copy
import numpy as np
from preprocessing.presetting import local_temp_directory

filename= "RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
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
genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
#genre_dtm_obj = genre_dtm_obj.eliminate(["periods"])
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_N = genre_dtm_obj.data_matrix_df
print("size of whole novellen sample: ", len(df_N))
print("Sample size Novellen (sample one instance per author): ", len(sample_n_from_cat(df_N)))

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[1]])
genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
#genre_dtm_obj = genre_dtm_obj.eliminate(["periods"])
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_E = genre_dtm_obj.data_matrix_df
print("Sample size Erzählungen (sample one instance per author): ", len(sample_n_from_cat(df_E)))


genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[2]])
genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_M = genre_dtm_obj.data_matrix_df
print("Sample size Märchen (sample one instance per author): ", len(sample_n_from_cat(df_M)))

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[3]])
genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_R = genre_dtm_obj.data_matrix_df
print("Sample size Romane (sample one instance per author): ", len(sample_n_from_cat(df_R)))

df_N_red = df_N.drop(columns=["Nachname"])
print("len df_N_red: ", len(df_N_red))
df_E_red = df_E.drop(columns=["Nachname"])
pairs_df = pairwise_self_counter_av_distances(sample_n_from_cat(df_R), sample_n_from_cat(df_M), metric)

pairs_dtm_obj = DTM(data_matrix_df=pairs_df, metadata_csv_filepath=metadata_path)
pairs_dtm_obj = pairs_dtm_obj.add_metadata(["Titel", "Nachname"])
print(pairs_dtm_obj.data_matrix_df.sort_values(by=["in_group_mean"]))



pairs_E_N_df = pairs_dtm_obj.data_matrix_df
pairs_E_N_df["pair_differences"] = pairs_E_N_df["out_group_mean"] - pairs_E_N_df["in_group_mean"]
pairs_E_N_df.loc["mean_pairwise_difference"] = pairs_E_N_df.mean()

pairs_E_N_df.to_csv(os.path.join(local_temp_directory(system), "pairs_rel_t-test_R-M.csv"))
print(pairs_E_N_df)

n = 1000
F_p_list = iterate_rel_ttest(n, df_N, df_E, select_one_author=True, metric=metric)
print("List of F and p values (iterated) between N vs E compared to N: ", F_p_list)
p_mean = np.array([e[1] for e in F_p_list]).mean()
print(p_mean)


F_p_list = iterate_rel_ttest(n, df_E, df_N, select_one_author=True, metric=metric)
print("List of F and p values (iterated) between E vs N compared to E: ", F_p_list)
p_mean = np.array([e[1] for e in F_p_list]).mean()
print(p_mean)

F_p_list = iterate_rel_ttest(n, df_R, df_M, select_one_author=True, metric=metric)
print("List of F and p values (iterated) between R vs M compared to R: ", F_p_list)
p_mean = np.array([e[1] for e in F_p_list]).mean()
print(p_mean)

from sklearn.model_selection import train_test_split
new_dtm_obj = copy(dtm_obj)
new_dtm_obj = new_dtm_obj.eliminate([year_cat, genre_cat, "Nachname", "periods"])
df = new_dtm_obj.data_matrix_df
rd_sample1, rd_sample2 = train_test_split(df, train_size=0.5)

F_p_list = random_iterate_rel_ttest(n, rd_sample1, rd_sample2, select_one_author=False, metric=metric, random_sample_size=150)
print(" List of randomized samples F, p: ", F_p_list)
p_mean = np.array([e[1] for e in F_p_list]).mean()
print(p_mean)


iter_obj = IterateDistanceCalc(n, df_N, df_E, metric=metric)
n_e_distances = iter_obj.distances

iter_obj = IterateDistanceCalc(10, df_N, metric=metric)
N_distances = iter_obj.distances

iter_obj = IterateDistanceCalc(10, df_E, metric=metric)
E_distances = iter_obj.distances

print(len(E_distances))

print("Für Novellen: ")
print("lower 5%: ", np.percentile(N_distances, 5))
print("upper 95%: ", np.percentile(N_distances, 95))
print("median: ", np.median(N_distances))
print("mean: ", np.mean(N_distances))

print("Für Erzählungen: ")
print("lower 5%: ", np.percentile(E_distances, 5))
print("upper 95%: ", np.percentile(E_distances, 95))
print("median: ", np.median(E_distances))
print("mean: ", np.mean(E_distances))

print("Für Erzählungen versus Novellen: ")
print("lower 5%: ", np.percentile(n_e_distances, 5))
print("upper 95%: ", np.percentile(n_e_distances, 95))
print("median: ", np.median(n_e_distances))
print("mean: ", np.mean(n_e_distances))
