from classification.perspectivalmodeling import sample_perspectival_sets, perspectival_sets_split, LR_perspectival_resample
from preprocessing.metadata_transformation import years_to_periods, generate_media_dependend_genres, standardize_meta_data_medium
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory
from preprocessing.presetting import load_stoplist
from preprocessing.corpus import DocFeatureMatrix
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

system = "wcph113"

infile_name = "red-to-2500mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" # "N-E_RFECV_red-to-213LRscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"

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

input_df = dtm_obj.data_matrix_df
print(input_df)


genre_labels = ["R", "E"]

#fit_df, transfer_df = sample_perspectival_sets(input_df, period_category="periods30a", metadata_cat_fit_list=["1820-1850"],
 #                            metadata_cat_transfer_list=["1850-1880"], genre_category="Gattungslabel_ED_normalisiert", genre_labels=genre_labels,
  #                           equal_sample_size_fit=True,equal_sample_size_transfer=True,
   #                          minor_frac_fit=1.0, minor_frac_transfer=1.0)

#print(fit_df)
#print(transfer_df)
#X_model_fit, X_model_transfer, Y_model_fit, Y_model_transfer = perspectival_sets_split(fit_df, transfer_df)

fit_period = ["1750-1850"]  # ["1820-1850"] # ["1850-1950"]
transfer_period = ["1850-1950"]  # ["1850-1880"] # ["1750-1850"]

fit_accuracy_scores_list, fit_f1scores_list, transfer_accuracy_scores_list, transfer_f1scores_list = LR_perspectival_resample(n=100, fit_val_size=0.3, input_df=input_df, period_category="periods100a", metadata_cat_fit_list=fit_period,
                             metadata_cat_transfer_list=transfer_period, genre_category="Gattungslabel_ED_normalisiert", genre_labels=genre_labels,
                             equal_sample_size_fit=True,equal_sample_size_transfer=True,
                             minor_frac_fit=1.0, minor_frac_transfer=1.0)

print(fit_accuracy_scores_list)

print(fit_f1scores_list)
print(transfer_accuracy_scores_list)
print(transfer_f1scores_list)

print("mean accuracy fit within", str(fit_period), ": ", np.asarray(fit_accuracy_scores_list).mean())
print("mean f1 scores fit within", str(fit_period), ": ", np.asarray(fit_f1scores_list).mean())
print("mean accuracy transfer to ",  str(transfer_period), ": ", np.asarray(transfer_accuracy_scores_list).mean())
print("mean f1 score transfer to ", str(transfer_period), ": ", np.asarray(transfer_f1scores_list).mean())
