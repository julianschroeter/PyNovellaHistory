from clustering.my_pca import PC_df
from preprocessing.presetting import vocab_lists_dicts_directory, local_temp_directory ,global_corpus_representation_directory, load_stoplist, global_corpus_raw_dtm_directory
from preprocessing.metadata_transformation import full_genre_labels, generate_media_dependend_genres
from preprocessing.corpus import DTM
from preprocessing.sampling import sample_n_from_cat
import os
import random
import pandas as pd

system = "wcph113" # "my_mac" # "wcph104" #  "my_xps"


colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "my_colors.txt"))
colors_list = random.sample(colors_list, len(colors_list))
colors_list = ["red", "green", "blue", "orange", "cyan", "yellow", "black", "magenta", "black", "brown", "lightblue"]
#colors_list = random.sample(colors_list, len(colors_list))

infilename = "red-to-1000mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" # "red-to-2500mfw_scaled_no-stopwords_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" #"raw_dtm_l1_lemmatized_use_idf_False2500mfw.csv" "red-to-100mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
infilename = "red-to-2500mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
dtm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), infilename)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

dtm_object = DTM(data_matrix_filepath =dtm_infile_path, metadata_csv_filepath=metadata_filepath)
dtm_object = dtm_object.add_metadata(["Gattungslabel_ED_normalisiert", "Medientyp_ED"])





labels_list = ["R", "M", "E", "N", "0E", "XE"]
labels_list = ["E", "N"]
labels_list = ["R", "M"]
labels_list = ["N", "E"]
labels_list = ["E", "N", "0E", "XE"]
dtm_object = dtm_object.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=labels_list)

input_df = dtm_object.data_matrix_df
# input_df = sample_n_from_cat(input_df)

replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "Erzählprosa",
                                    "R": "Roman", "M": "Märchen", "XE": "Erzählprosa"}}

replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "N", "E": "E", "0E": "E",
                                     "XE": "E"}}

input_df = full_genre_labels(input_df, replace_dict=replace_dict)


med_dep_df = generate_media_dependend_genres(input_df)

med_dep_df = med_dep_df.drop(columns=["Gattungslabel_ED_normalisiert", "Medientyp_ED"])

dep_labels_list = ["Familienblatt_Novelle", "Familienblatt_Erzählung", "Rundschau_Erzählung", "Rundschau_Novelle"]
med_dep_df = med_dep_df[med_dep_df.isin({"dependent_genre": dep_labels_list}).any(1)]

print(med_dep_df)

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

dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1750, end_year=1951, epoch_length=100,
                                          new_periods_column_name="periods")
dtm_obj = dtm_obj.eliminate([year_cat])

# use just <1850 for anthologies
dtm_obj.data_matrix_df = dtm_obj.data_matrix_df[dtm_obj.data_matrix_df["periods"] == "1750-1850"]

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

list_of_med_dep_df = [dfs_list[1] for dfs_list in list_of_genre_dfs if dfs_list[0] in ["Anthologie-Erzählung", "Anthologie-Novelle", "Anthologie-MLP",
                                                                                       "Taschenbuch-Novelle", "Taschenbuch-Erzählung", "Taschenbuch-MLP"] ]
med_dep_df = pd.concat(list_of_med_dep_df)
med_dep_df = med_dep_df.drop(columns=[medium_cat, name_cat, "periods"])

print(med_dep_df)

pc_df = PC_df(input_df=med_dep_df)

pc_df.generate_pc_df(n_components=0.95)

target_df = pc_df.pc_target_df

# drop outliers
pc_df.pc_target_df =  pc_df.pc_target_df.drop(pc_df.pc_target_df["PC_1"].idxmax())
pc_df.pc_target_df = pc_df.pc_target_df.drop(pc_df.pc_target_df["PC_2"].idxmax())

print(pc_df.pc_target_df.sort_values(by=["PC_2"], axis=0, ascending=False))

print(pc_df.component_loading_df.iloc[0, :].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[1, :].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[0,:].sort_values(ascending=True)[:20])
print(pc_df.component_loading_df.loc[1,: ].sort_values(ascending=True)[:20])
print(pc_df.pca.explained_variance_)
pc_df.scatter(colors_list)

import matplotlib.pyplot as plt
plt.savefig(os.path.join(local_temp_directory(system), "PCA_med_dep_Rundschau.png"))