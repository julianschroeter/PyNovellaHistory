system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')



from preprocessing.distances import IterateDistanceCalc, pairwise_self_counter_av_distances, GroupDistances, InterGroupDistances, random_iterate_rel_ttest, iterate_rel_ttest
from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory, local_temp_directory
from preprocessing.metadata_transformation import full_genre_labels, years_to_periods
from preprocessing.sampling import split_to2samples, sample_n_from_cat

import os
import matplotlib.pyplot as plt
from preprocessing.presetting import local_temp_directory
import pandas as pd


filename= "RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"

metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

label_list = ["R", "M", "E", "N", "0E", "XE"]
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"
name_cat = "Nachname"

metric = "cosine"

dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)
dtm_obj = dtm_obj.add_metadata([genre_cat, year_cat, name_cat])
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge", "de", "di"])

dtm_obj.data_matrix_df = dtm_obj.data_matrix_df.drop(["00475-00", "00349-00", "00490-00", "00580-00"]) # remove Dubletten and texts that are intentionally represented twice in the corpus

dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1790, end_year=1950, epoch_length=1,
                                          new_periods_column_name="periods")
dtm_obj = dtm_obj.eliminate([year_cat])

label_list = ["N", "E", "M", "R"]
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[0]])
df_N = genre_dtm_obj.data_matrix_df
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[1]])
df_E = genre_dtm_obj.data_matrix_df
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[2]])
df_M = genre_dtm_obj.data_matrix_df
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[3]])
df_R = genre_dtm_obj.data_matrix_df


bool_select_one_author = True
bool_select_one_per_period = False

R_dist_obj = GroupDistances(df_R, metric, select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period)
M_dist_obj = GroupDistances(df_M, metric, select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period)
E_dist_obj = GroupDistances(df_E, metric, select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period)
N_dist_obj = GroupDistances(df_N, metric, select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period)


distance_matrix_N_intra = N_dist_obj.dist_matr_df
print(distance_matrix_N_intra)
distance_matrix_N_intra.to_csv(os.path.join(local_temp_directory(system), "N_intra_dist_matrix.csv"))

N_E_inter_dist_obj = InterGroupDistances(df_N, df_E, metric,select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period, smaller_sample_size=False)
N_E_dist_matr = N_E_inter_dist_obj.dist_matr_df
N_E_dist_matr.to_csv(os.path.join(local_temp_directory(system), "N_E_inter_dist_matrix.csv"))