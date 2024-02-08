import pandas as pd

system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')



from metrics.distances import DistResults, GroupDistances, results_2groups_dist
from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory, local_temp_directory
from preprocessing.presetting import local_temp_directory

from preprocessing.metadata_transformation import years_to_periods

import os
from sklearn.model_selection import train_test_split
from itertools import combinations
import json


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

dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1770, end_year=1950, epoch_length=10,
                                          new_periods_column_name="periods")
dtm_obj = dtm_obj.eliminate([year_cat])

list_of_genre_dfs = []

label_list = ["N", "E", "0E", "XE", "M"]
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=label_list)

all_df = genre_dtm_obj.data_matrix_df

grouped = all_df.groupby("periods")
periods_dfs = [grouped.get_group(df) for df in grouped.groups]

list_of_periods_dfs = []
for df in periods_dfs:
    period_value = df.iloc[0]["periods"]
    list_of_periods_dfs.append([str(period_value), df])

periods_list = [e[0] for e in list_of_periods_dfs]

for pair in combinations(list_of_periods_dfs,2):
    a, b = pair
    start_a, start_b = a[0], b[0]
    start_a, start_b = int(start_a[:4]), int(start_b[:4])
    if abs(start_b - start_a) >= 40:
        distances = results_2groups_dist(10, b[1], a[1], select_one_author=False,select_one_per_period=False, metric="cosine")
        print(f"Wähle aus Periode {b[0]} den Text aus, der zu den Texten aus Period {a[0]} die größte Distanz hat:")
        print("mittlere Cosinus-Gruppendistanz (und std) für die Texte des zweiten Zeitraums, Samplegrößen, Texte mit min. /max. Distanz aus der ersten Gruppe zur zweiten: ", distances)
        print("Text mit geringster Distanz")
        print(metatdata_df.loc[distances["instance_of_df1_with_min_dist_to_df2_group"][0][0],["Nachname", "Titel", "Jahr_ED", "Gattungslabel_ED_normalisiert"]])
        print(distances["instance_of_df1_with_min_dist_to_df2_group"][0][1][1])

        print("Text mit maximaler Distanz:")
        print(metatdata_df.loc[
                  distances["instance_of_df1_with_max_dist_to_df2_group"][0][0], ["Nachname", "Titel", "Jahr_ED", "Gattungslabel_ED_normalisiert"]])
        print(distances["instance_of_df1_with_max_dist_to_df2_group"][0][1][1])
