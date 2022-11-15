from classification.perspectivalmodeling import sample_perspectival_sets, perspectival_sets_split, LR_perspectival_resample
from preprocessing.metadata_transformation import years_to_periods, generate_media_dependend_genres, standardize_meta_data_medium
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory
from preprocessing.presetting import load_stoplist
from preprocessing.corpus_alt import DocFeatureMatrix
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from clustering.my_pca import PC_df

system = "wcph113"

infile_name =  "N-E_RFECV_red-to-213LRscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" #"red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" #

dtm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system),infile_name)
metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

dtm_obj = DocFeatureMatrix(data_matrix_filepath=dtm_infile_path, metadata_csv_filepath=metadata_path)

dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED"])
dtm_obj.data_matrix_df = years_to_periods(input_df=dtm_obj.data_matrix_df, category_name="Jahr_ED",
                                         start_year=1750, end_year=1951, epoch_length=100,
                                          new_periods_column_name="periods100a")

#dtm_obj.data_matrix_df = years_to_periods(input_df=dtm_obj.data_matrix_df, category_name="Jahr_ED",
 #                                         start_year=1790, end_year=1951, epoch_length=30,
  #                                        new_periods_column_name="periods30a")

dtm_obj = dtm_obj.eliminate(["Jahr_ED"])
genre_labels = ["N", "E"]
dtm_obj = dtm_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", genre_labels)
df = dtm_obj.data_matrix_df

colors_list = ["blue", "royalblue", "darkgreen", "green"]
print(df)

df["period_dep_genre"] = df.apply(lambda x: str(x["Gattungslabel_ED_normalisiert"]) + "_" + str(x["periods100a"]), axis=1)

df = df.drop(columns=["Gattungslabel_ED_normalisiert", "periods100a"])
print(df)


pc_df = PC_df(input_df=df)

pc_df.generate_pc_df(n_components=0.95)


print(pc_df.pc_target_df.sort_values(by=["PC_2"], axis=0, ascending=False))

print(pc_df.component_loading_df.iloc[0, :].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[1, :].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[0,:].sort_values(ascending=True)[:20])
print(pc_df.component_loading_df.loc[1,: ].sort_values(ascending=True)[:20])
print(pc_df.pca.explained_variance_)
pc_df.scatter(colors_list)

