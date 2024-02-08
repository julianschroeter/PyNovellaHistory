import pandas as pd

system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')



from metrics.distances import DistResults, GroupDistances, results_2groups_dist, results_1group_dist
from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory, local_temp_directory
from preprocessing.presetting import local_temp_directory

from preprocessing.metadata_transformation import years_to_periods

import os
from sklearn.model_selection import train_test_split
from itertools import combinations
from collections import Counter


filename = "red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename_base = os.path.splitext(filename)[0]

metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

label_list = ["R", "M", "E", "N", "0E", "XE"]
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"
name_cat = "Nachname"

metric = "cosine"
n=100

outfile_csv_path = str(n) + "_" + metric + "_distances_precursors_" + filename
outfile_dict_txt_path = str(n) + "_" + metric + "_distances_precursors" + filename_base + ".txt"
outfile_results_df_path = os.path.join(local_temp_directory(system), outfile_csv_path)
outfile_dict_path = os.path.join(local_temp_directory(system), outfile_dict_txt_path)

metatdata_df = pd.read_csv(metadata_path, index_col=0)

dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)
dtm_obj = dtm_obj.add_metadata([genre_cat, year_cat, name_cat])
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge", "de", "di"])

dtm_obj.data_matrix_df = dtm_obj.data_matrix_df.drop(["00475-00", "00349-00", "00490-00", "00580-00"]) # remove Dubletten and texts that are intentionally represented twice in the corpus

list_of_farest_contemporaries = []
for i in range(20):
    red_dtm_obj = dtm_obj
    red_dtm_obj.data_matrix_df = years_to_periods(red_dtm_obj.data_matrix_df,category_name=year_cat, start_year=1765+i, end_year=1950, epoch_length=20,
                                              new_periods_column_name="periods")
    red_dtm_obj = red_dtm_obj.eliminate([year_cat])

    list_of_genre_dfs = []

    label_list = ["N", "E", "0E", "XE", "M"]
    genre_dtm_obj = red_dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=label_list)
    all_df = genre_dtm_obj.data_matrix_df

    grouped = all_df.groupby("periods")
    periods_dfs = [grouped.get_group(df) for df in grouped.groups]

    list_of_periods_dfs = []
    for df in periods_dfs:
        period_value = df.iloc[0]["periods"]
        list_of_periods_dfs.append([str(period_value), df])

    periods_list = [e[0] for e in list_of_periods_dfs]

    for period in periods_list:
        for sub_period, df in list_of_periods_dfs:
            if sub_period == period:
                distances = results_1group_dist(n=10, input_df=df, select_one_per_period=False, select_one_author=False,metric="cosine")

                print("max: ", distances["max_dist"])
                print("min: ", distances["min_dist"])
                list_of_farest_contemporaries.append(distances["max_dist"][0][0])




print(list_of_farest_contemporaries)
nearest_precursors = Counter(list_of_farest_contemporaries)
print(nearest_precursors)
nearest_prec = pd.DataFrame.from_dict(nearest_precursors, orient="index")

precursors = [key for key, value in nearest_precursors.most_common()]

prec_df = metatdata_df.loc[precursors,["Nachname", "Titel", "Jahr_ED", "Gattungslabel_ED_normalisiert", "Medientyp_ED"]]

new_df = pd.concat([nearest_prec,prec_df], axis=1)
print(new_df)

new_df.to_csv(os.path.join(local_temp_directory(system), "farest_contemporaries.csv"))