system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')



from metrics.distances import DistResults
from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory
from preprocessing.metadata_transformation import years_to_periods

import os
from preprocessing.presetting import local_temp_directory
import json




filename= "RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-2500mfw_red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"

metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)



label_list = ["N", "E", "0E", "XE"]
author_cat = "Nachname"
year_cat = "Jahr_ED"
dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)
genre_cat = "Gattungslabel_ED_normalisiert"
dtm_obj = dtm_obj.add_metadata([year_cat, genre_cat, author_cat])
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge", "de", "di"])

dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1790, end_year=1930, epoch_length=30,
                                          new_periods_column_name="periods30a")

dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1790, end_year=1930, epoch_length=1,
                                          new_periods_column_name="periods")

dtm_obj = dtm_obj.eliminate([year_cat])


dtm_obj_red = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=label_list)
#dtm_obj_red = dtm_obj_red.eliminate([genre_cat, year_cat])
df = dtm_obj_red.data_matrix_df

all_df = df.drop(columns="periods30a")
print(all_df)

grouped = df.groupby("periods30a")
periods_dfs = [grouped.get_group(df) for df in grouped.groups]

list_of_periods_dfs = []
for df in periods_dfs:
    period_value = df.iloc[0]["periods30a"]
    print(period_value)
    df = df.drop(columns="periods30a")
    print(df.columns[-5:])
    list_of_periods_dfs.append([str(period_value), df])

print(list_of_periods_dfs)

n = 100

metric = "euclidean"
metric = "cityblock"
metric = "cosine"

outfile_df_name = str(n) + "_" + metric + "periods-distance_results_6000mfw.csv"
outfile_dict_name = str(n) + "N" + metric + "periods-min_max_distances_dict_6000mfw.txt"
outfile_results_df_path = os.path.join(local_temp_directory(system), outfile_df_name)


outfile_dict_path = os.path.join(local_temp_directory(system), outfile_dict_name)

min_max_results_dict = {}


dist_results_obj = DistResults(n, input_df_1=all_df, input_df_2=None, metric=metric, label1="all", label2=None, select_one_author=True)

for label, df in list_of_periods_dfs:
    dist_results_obj.add_result(n, input_df_1=df, label1=label, metric=metric)

from itertools import combinations, permutations

genre_df_perm = combinations(list_of_periods_dfs, 2)

for name_df_comb in genre_df_perm:
    label1 = name_df_comb[0][0]
    label2 = name_df_comb[1][0]
    df_1 = name_df_comb[0][1]
    df_2 = name_df_comb[1][1]
    dist_results_obj.add_result(n, input_df_1=df_1, input_df_2=df_2, label1=label1, label2=label2, metric=metric)

dist_results_obj.calculate_differences_of_distances()
dist_results_obj.calculate_ratio_of_distances()

results_df = dist_results_obj.results_df
#results_df.columns = pd.MultiIndex.from_tuples([(metric, "mean"), (metric, "mean_of_stds")])

results_df.to_csv(path_or_buf=outfile_results_df_path)

with open(outfile_dict_path, 'w') as file:
    file.write(json.dumps(dist_results_obj.min_max_results_dict))  # use `json.loads` to do the reverse