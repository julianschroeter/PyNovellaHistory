system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')



from metrics.distances import IterateDistanceCalc, results_1group_dist, results_2groups_dist, DistResults
from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory
from preprocessing.metadata_transformation import full_genre_labels, years_to_periods
from preprocessing.sampling import split_to2samples
from preprocessing.util import first_n_dict_entries, from_n_dict_entries

import os
from copy import copy
import pandas as pd
from preprocessing.presetting import local_temp_directory
import json



filename= "RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-2500mfw_red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-2500mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-2500mfw_red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename_base = os.path.splitext(filename)[0]


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

#dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1750, end_year=1950, epoch_length=100,
 #                                         new_periods_column_name="periods100a")

dtm_obj = dtm_obj.eliminate([year_cat])

list_of_author_dfs = []

names_list = ["Eichendorff", "Büchner", "Goethe", "Tieck", "Kleist", "Storm", "Keller", "Chamisso",
              "Häring", "Raabe", "John", "Fontane", "Musil", "Schnitzler", "Zweig", "Mann", "Roth"]

label_list = ["N", "E", "0E", "XE"]
dtm_obj_red = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=label_list)


periods_list = ["1750-1850", "1850-1950"]

#before_df, after_df = split_to2samples(dtm_obj_red.data_matrix_df, metadata_category="periods100a", label_list=periods_list)

n = 1

outfile_df_name = str(n) + "_" + metric + "selectet_author-distance_results_6000mfw.csv"
outfile_dict_name = str(n) + "N" + metric + "selectet_author-min_max_distances_dict_6000mfw.txt"
outfile_results_df_path = os.path.join(local_temp_directory(system), outfile_df_name)

print("proceed: " + outfile_df_name)

outfile_dict_path = os.path.join(local_temp_directory(system), outfile_dict_name)

min_max_results_dict = {}

bool_select_one_author = False
bool_select_one_per_period =False

#df = before_df
df_all = dtm_obj_red.data_matrix_df

dist_results_obj = DistResults(n, input_df_1=df_all, input_df_2=None, metric=metric, label1="all", label2=None,
                               select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period)

list_of_author_dfs = []



new_dtm_obj = DTM(data_matrix_df=df_all, metadata_csv_filepath=metadata_path)
#new_dtm_obj = new_dtm_obj.add_metadata([name_cat, genre_cat])

label_list = ["Goethe"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])
label_list = ["Tieck"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])


label_list = ["Wieland"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])



label_list = ["Eichendorff"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])



label_list = ["Storm"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])

label_list = ["Kleist"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])
print(list_of_author_dfs)


label_list = ["Büchner"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])

label_list = ["Keller"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])

label_list = ["Häring"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])

label_list = ["Raabe"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])

label_list = ["John"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])#

label_list = ["Fontane"]
new_dtm_obj_genre = new_dtm_obj.reduce_to_categories(metadata_category=name_cat, label_list=label_list)
df = new_dtm_obj_genre.data_matrix_df
list_of_author_dfs.append([label_list, df])

for label_list, df in list_of_author_dfs:
    print(df)
    dist_results_obj.add_result(n, input_df_1=df, label1=str(label_list[0]), metric=metric)

from itertools import combinations, permutations

genre_df_perm = combinations(list_of_author_dfs, 2)

for name_df_comb in genre_df_perm:
    label1 = "_".join(name_df_comb[0][0])
    label2 = "_".join(name_df_comb[1][0])
    df_1 = name_df_comb[0][1]
    df_2 = name_df_comb[1][1]
    dist_results_obj.add_result(n, input_df_1=df_1, input_df_2=df_2, label1=label1, label2=label2, metric=metric)

dist_results_obj.calculate_differences_of_distances()
dist_results_obj.calculate_ratio_of_distances()

results_df = dist_results_obj.results_df
print(results_df)
#results_df.columns = pd.MultiIndex.from_tuples([(metric, "mean"), (metric, "mean_of_stds")])

results_df.to_csv(path_or_buf=outfile_results_df_path)

with open(outfile_dict_path, 'w') as file:
    file.write(json.dumps(dist_results_obj.min_max_results_dict))  # use `json.loads` to do the reverse