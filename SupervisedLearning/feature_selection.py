from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
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

dtm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), "dtm_lemma_tfidf10000semantic_periods100a.csv")

input_df = pd.read_csv(dtm_infile_path, index_col=0)


fit_df, transfer_df = sample_perspectival_sets(input_df, period_category="periods100a", metadata_cat_fit_list=["1750-1850"],
                             metadata_cat_transfer_list=["1850-1950"], genre_category="Gattungslabel_ED", genre_labels=["N", "0E"],
                             equal_sample_size_fit=True,equal_sample_size_transfer=True,
                             minor_frac_fit=1.0, minor_frac_transfer=1.0)

print(fit_df)
print(transfer_df)
X_model_fit, X_model_transfer, Y_model_fit, Y_model_transfer = perspectival_sets_split(fit_df, transfer_df)

min_features_to_select = 1
estimator = LogisticRegression(solver="lbfgs", penalty="l2")
selector = RFECV(estimator, step=10, min_features_to_select=min_features_to_select, cv=5)
selector.fit(X_model_fit, Y_model_fit)
supported_features = selector.support_
print(supported_features)
supported_features = supported_features.tolist()

dropped_input_df = input_df.drop(columns=["Gattungslabel_ED", "periods100a"])
new_df = dropped_input_df.loc[:, supported_features]
print(new_df)

print("Optimal number of features : %d" % selector.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(range(min_features_to_select,
               len(selector.grid_scores_) + min_features_to_select),
         selector.grid_scores_)
plt.show()