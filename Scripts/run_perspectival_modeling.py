from SupervisedLearning.PerspectivalModeling import sample_perspectival_sets, perspectival_sets_split, LR_perspectival_resample
from Preprocessing.MetadataTransformation import years_to_periods, generate_media_dependend_genres, standardize_meta_data_medium
from Preprocessing.Presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory
from Preprocessing.Presetting import load_stoplist
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

system = "my_mac"

dtm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), "dtm_lemma_tfidf10000_periods100a.csv")

input_df = pd.read_csv(dtm_infile_path, index_col=0)


fit_df, transfer_df = sample_perspectival_sets(input_df, period_category="periods100a", metadata_cat_fit_list=["1750-1850"],
                             metadata_cat_transfer_list=["1850-1950"], genre_category="Gattungslabel_ED", genre_labels=["N", "0E"],
                             equal_sample_size_fit=True,equal_sample_size_transfer=True,
                             minor_frac_fit=1.0, minor_frac_transfer=1.0)

print(fit_df)
print(transfer_df)
X_model_fit, X_model_transfer, Y_model_fit, Y_model_transfer = perspectival_sets_split(fit_df, transfer_df)

fit_accuracy_scores_list, fit_f1scores_list, transfer_accuracy_scores_list, transfer_f1scores_list = LR_perspectival_resample(n=100, fit_val_size=0.1, input_df=input_df, period_category="periods100a", metadata_cat_fit_list=["1750-1850"],
                             metadata_cat_transfer_list=["1850-1950"], genre_category="Gattungslabel_ED", genre_labels=["N", "0E"],
                             equal_sample_size_fit=True,equal_sample_size_transfer=True,
                             minor_frac_fit=1.0, minor_frac_transfer=1.0)

print(fit_accuracy_scores_list)

print(fit_f1scores_list)
print(transfer_accuracy_scores_list)
print(transfer_f1scores_list)

print("mean accuracy fit within 1750-1850:", np.asarray(fit_accuracy_scores_list).mean())
print("mean f1 scores fit within 1750-1850:", np.asarray(fit_f1scores_list).mean())
print("mean accuracy transfer to 1850-1950:", np.asarray(transfer_accuracy_scores_list).mean())
print("mean f1 score transfer to 1850-1950:", np.asarray(transfer_f1scores_list).mean())
