from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from classification.perspectivalmodeling import sample_perspectival_sets, perspectival_sets_split
from preprocessing.presetting import local_temp_directory, global_corpus_raw_dtm_directory
import os
import matplotlib.pyplot as plt
import pandas as pd

system = "wcph113" # "my_mac"

dtm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), "dtm_lemma_tfidf10000semantic_periods100a.csv")
new_df_outfile_path = os.path.join(local_temp_directory(system), "RFE_dtm_lemma_tfidfsemantic_before1850.csv")
fig_filename = os.path.join(local_temp_directory(system), "RFE_plot.png")
input_df = pd.read_csv(dtm_infile_path, index_col=0)


fit_df, transfer_df = sample_perspectival_sets(input_df, period_category="periods100a", metadata_cat_fit_list=["1750-1850"],
                             metadata_cat_transfer_list=["1850-1950"], genre_category="Gattungslabel_ED", genre_labels=["N", "E"],
                             equal_sample_size_fit=True,equal_sample_size_transfer=True,
                             minor_frac_fit=1.0, minor_frac_transfer=1.0)

print(fit_df)
print(transfer_df)
X_model_fit, X_model_transfer, Y_model_fit, Y_model_transfer = perspectival_sets_split(fit_df, transfer_df)

min_features_to_select = 1
estimator = LogisticRegression(solver="lbfgs", penalty="l2", multi_class="ovr")
estimator = SVC(kernel="linear")
selector = RFECV(estimator, step=10, min_features_to_select=min_features_to_select, cv=5)

selector.fit(X_model_fit, Y_model_fit)
supported_features = selector.support_
print(supported_features)
supported_features = supported_features.tolist()

dropped_input_df = input_df.drop(columns=["Gattungslabel_ED", "periods100a"])
new_df = dropped_input_df.loc[:, supported_features]
print(new_df)

new_df.to_csv(path_or_buf= new_df_outfile_path)
print("Optimal number of features : %d" % selector.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()

plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(range(min_features_to_select,
               len(selector.grid_scores_) + min_features_to_select),
         selector.grid_scores_)
plt.show()
plt.savefig(fig_filename)