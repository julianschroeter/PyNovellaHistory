system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')



from metrics.distances import DistResults, GroupDistances
from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory, local_temp_directory
from preprocessing.presetting import local_temp_directory

from preprocessing.metadata_transformation import years_to_periods

import os
from sklearn.model_selection import train_test_split
from itertools import combinations
import json






filename = "scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-1000mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-500mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename= "RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-100mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-100mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-1000mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_no-stopwords_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "red-to-2500mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"

filename_base = os.path.splitext(filename)[0]


metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

label_list = ["R", "M", "E", "N", "0E", "XE"]
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"
name_cat = "Nachname"
medium_cat = "Medientyp_ED"

metric = "cosine"
n=100

outfile_csv_path = str(n) + "_" + metric + "_distances_media-dependent-genres_" + filename
outfile_dict_txt_path = str(n) + "_" + metric + "_distances-media-dependent-genres_" + filename_base + ".txt"
outfile_results_df_path = os.path.join(local_temp_directory(system), outfile_csv_path)
outfile_dict_path = os.path.join(local_temp_directory(system), outfile_dict_txt_path)

dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)
dtm_obj = dtm_obj.add_metadata([genre_cat, year_cat, name_cat, medium_cat])
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge", "de", "di"])

dtm_obj.data_matrix_df = dtm_obj.data_matrix_df.drop(["00475-00", "00349-00", "00490-00", "00580-00"]) # remove Dubletten and texts that are intentionally represented twice in the corpus

dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1750, end_year=1950, epoch_length=100,
                                          new_periods_column_name="periods")
dtm_obj = dtm_obj.eliminate([year_cat])

list_of_genre_dfs = []

label_list = ["N", "E", "M", "R", "0E", "XE"]
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[0]])

df_N = genre_dtm_obj.data_matrix_df
#list_of_genre_dfs.append([label_list[0], df_N])

df_N_TB = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Taschenbuch"]).data_matrix_df
df_N_TB.loc[:,genre_cat] = "Taschenbuch-Novelle"
df_N_Anth = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Anthologie"]).data_matrix_df
df_N_Anth.loc[:,genre_cat] = "Anthologie-Novelle"
df_N_FamBl = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Familienblatt"]).data_matrix_df
df_N_FamBl.loc[:,genre_cat] = "Familienblatt-Novelle"
df_N_Rundsch = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Rundschau"]).data_matrix_df
df_N_Rundsch.loc[:,genre_cat] = "Rundschau-Novelle"

list_of_genre_dfs.append(["Taschenbuch-Novelle", df_N_TB])
list_of_genre_dfs.append(["Anthologie-Novelle", df_N_Anth])
list_of_genre_dfs.append(["Familienblatt-Novelle", df_N_FamBl])
list_of_genre_dfs.append(["Rundschau-Novelle", df_N_Rundsch])



genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[1]])
df_E = genre_dtm_obj.data_matrix_df
#list_of_genre_dfs.append([label_list[1], df_E])

df_E_TB = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Taschenbuch"]).data_matrix_df
df_E_TB.loc[:,genre_cat] = "Taschenbuch-Erzählung"
df_E_Anth = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Anthologie"]).data_matrix_df
df_E_Anth.loc[:,genre_cat] = "Anthologie-Erzählung"
df_E_FamBl = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Familienblatt"]).data_matrix_df
df_E_FamBl.loc[:,genre_cat] = "Familienblatt-Erzählung"
df_E_Rundsch = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Rundschau"]).data_matrix_df
df_E_Rundsch.loc[:,genre_cat] = "Rundschau-Erzählung"
list_of_genre_dfs.append(["Taschenbuch-Erzählung", df_E_TB])
list_of_genre_dfs.append(["Anthologie-Erzählung", df_E_Anth])
list_of_genre_dfs.append(["Familienblatt-Erzählung", df_E_FamBl])
# list_of_genre_dfs.append(["Rundschau-Erzählung", df_E_Rundsch])

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[4], label_list[5]])
df_0XE = genre_dtm_obj.data_matrix_df
#list_of_genre_dfs.append(["0XE", df_0XE])

df_0XE_TB = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Taschenbuch"]).data_matrix_df
df_0XE_TB.loc[:,genre_cat] = "Taschenbuch-MLP"
df_0XE_Anth = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Anthologie"]).data_matrix_df
df_0XE_Anth.loc[:,genre_cat] = "Anthologie-MLP"
df_0XE_FamBl = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Familienblatt"]).data_matrix_df
df_0XE_FamBl.loc[:,genre_cat] = "Familienblatt-MLP"
df_0XE_Rundsch = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Rundschau"]).data_matrix_df
df_0XE_Rundsch.loc[:,genre_cat] = "Rundschau-MLP"
list_of_genre_dfs.append(["Taschenbuch-MLP", df_0XE_TB])
list_of_genre_dfs.append(["Anthologie-MLP", df_0XE_Anth])
list_of_genre_dfs.append(["Familienblatt-MLP", df_0XE_FamBl])

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[1], label_list[4], label_list[5]])
df_0XE_E_Rundsch = genre_dtm_obj.reduce_to_categories(metadata_category=medium_cat, label_list=["Rundschau"]).data_matrix_df
list_of_genre_dfs.append(["Rundschau-MLP+Erzählung", df_0XE_Rundsch])



genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[2]])
#df_M = genre_dtm_obj.data_matrix_df
#list_of_genre_dfs.append([label_list[2], df_M])
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[3]])
#df_R = genre_dtm_obj.data_matrix_df
#list_of_genre_dfs.append([label_list[3], df_R])

#genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[5]])
#df_XE = genre_dtm_obj.data_matrix_df
#list_of_genre_dfs.append([label_list[5], df_XE])

bool_select_one_author = True
bool_select_one_per_period = False

rd_sample_size = 150
all_df = dtm_obj.data_matrix_df
all_df = all_df.drop(columns=[medium_cat])

#rd_sample1, rd_sample2 = train_test_split(all_df.sample(frac=1.0), train_size=0.5)
#rd_sample1.loc[:,genre_cat] = "Z1"
#rd_sample2.loc[:,genre_cat] = "Z2"

#list_of_genre_dfs.append(["rdsample1", rd_sample1])
#list_of_genre_dfs.append(["rdsample2", rd_sample2])


dist_results_obj = DistResults(n, input_df_1=all_df, input_df_2=None, metric=metric, label1="all", label2=None,
                               select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period,
                               smaller_sample_size=False, sample_size_df_1=None, sample_size_df_2=None)

for label, df in list_of_genre_dfs:

    if "rdsample" in label:
        dist_results_obj.add_result(n, input_df_1=df, label1= label, metric=metric,
                                select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period,
                                sample_size_df_1=None)
    else:
        print(label, df)
        df = df.drop(columns=[medium_cat])
        print(label, df)
        dist_results_obj.add_result(n, input_df_1=df, label1=label, metric=metric,
                                    select_one_author=bool_select_one_author,
                                    select_one_per_period=bool_select_one_per_period,
                                    sample_size_df_1=None)

genre_df_comb = combinations(list_of_genre_dfs, 2)

print(list_of_genre_dfs)
print(genre_df_comb)

for name_df_comb in genre_df_comb:
    label1 = str(name_df_comb[0][0])
    print(label1)
    label2 = str(name_df_comb[1][0])
    print(label2)
    df_1 = name_df_comb[0][1]
    if medium_cat in df_1.columns:
        df_1 = df_1.drop(columns=[medium_cat])
    df_2 = name_df_comb[1][1]
    if medium_cat in df_2.columns:
        df_2 = df_2.drop(columns=[medium_cat])
    if "rdsample" in label1 and "rdsample" not in label2:

        dist_results_obj.add_result(n, input_df_1=df_1, input_df_2=df_2, label1=label1, label2=label2, metric=metric,
                                    select_one_author=bool_select_one_author, select_one_per_period=bool_select_one_per_period,
                                    smaller_sample_size=True, sample_size_df_1=None, sample_size_df_2=None)
    elif "rdsample" not in label1 and "rdsample" in label2:

        dist_results_obj.add_result(n, input_df_1=df_1, input_df_2=df_2, label1=label1, label2=label2, metric=metric,
                                    select_one_author=bool_select_one_author,
                                    select_one_per_period=bool_select_one_per_period,
                                    smaller_sample_size=True, sample_size_df_1=None, sample_size_df_2=None)
    elif "rdsample" in label1 and "rdsample" in label2:

        dist_results_obj.add_result(n, input_df_1=df_1, input_df_2=df_2, label1=label1, label2=label2, metric=metric,
                                    select_one_author=bool_select_one_author,
                                    select_one_per_period=bool_select_one_per_period,
                                    smaller_sample_size=False, sample_size_df_1=None,sample_size_df_2=None)
    else:
        print("nächste 2 dfs: ")
        print(df_1, df_2)
        dist_results_obj.add_result(n, input_df_1=df_1, input_df_2=df_2, label1=label1, label2=label2, metric=metric,
                                    select_one_author=bool_select_one_author,
                                    select_one_per_period=bool_select_one_per_period,
                                    smaller_sample_size=False, sample_size_df_1=None,
                                    sample_size_df_2=None)


print(dist_results_obj.results_df)

dist_results_obj.calculate_differences_of_distances()
dist_results_obj.calculate_ratio_of_distances()

results_df = dist_results_obj.results_df
#results_df.columns = pd.MultiIndex.from_tuples([(metric, "mean"), (metric, "mean_of_stds")]) # if multilevel columns are used

results_df.to_csv(path_or_buf=outfile_results_df_path)

with open(outfile_dict_path, 'w') as file:
    file.write(json.dumps(dist_results_obj.min_max_results_dict))  # use `json.loads` to do the reverse