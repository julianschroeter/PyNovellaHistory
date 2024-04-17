system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# from my own modules:
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods

# standard libraries
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import pymc as pm
from pymc import HalfCauchy, Model, Normal
import arviz as az
import bambi as bmb
from scipy.stats import chi2_contingency
from numpy import cov
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"



old_infile_name = os.path.join(global_corpus_representation_directory(system), "SNA_novellas.csv")
infile_name = os.path.join(global_corpus_representation_directory(system), "Network_Matrix_all.csv")

infile_name = os.path.join(global_corpus_representation_directory(system), "conll_based_networkdata-matrix-novellas_15mostcommon.csv")
print(infile_name)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)
#matrix_obj = matrix_obj.add_metadata([genre_cat, year_cat, medium_cat, "Nachname", "Titel"])

df1 = matrix_obj.data_matrix_df

matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix_obj.data_matrix_df, metadata_csv_filepath=old_infile_name)
#matrix_obj = matrix_obj.add_metadata(["Netzwerkdichte"])

length_infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix_obj.data_matrix_df,
                              metadata_csv_filepath=length_infile_df_path)
matrix_obj = matrix_obj.add_metadata(["token_count"])


cat_labels = ["N", "E"]
cat_labels = ["N", "E", "0E", "XE", "R"]
matrix_obj = matrix_obj.reduce_to_categories(genre_cat, cat_labels)

matrix_obj = matrix_obj.eliminate(["Figuren"])

df = matrix_obj.data_matrix_df

df.rename(columns={"scaled_centralization_conll": "dep_var"}, inplace=True) # "Netzwerkdichte"

#df = df[~df["token_count"].isna()]
#df = df[~df["Netzwerkdichte_conll"].isna()]
#df = df[~df["Zentralisierung"].isna()]
#df = df[~df["dep_var"].isna()]


dep_var = "dep_var"
df.rename(columns={"Zentralisierung": "dep_var",
                   "Netzwerkdichte_conll":"Netzwerkdichte"}, inplace=True) # "Netzwerkdichte"
#df["dep_var"] = df["dep_var"].astype(float)
df.dropna(subset = ['token_count', 'Netzwerkdichte', "dep_var"], inplace=True)


print(df["token_count"])
print(df["Netzwerkdichte"])
print(len(df["dep_var"]))
#
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
axes[0].scatter(df['dep_var'], df['token_count'], color="grey")
axes[0].set_title("Zentralisierung auf Textumfang")
axes[0].set_xlabel("Zentralisierung")
axes[1].scatter(df['Netzwerkdichte'], df['token_count'], color="black")
axes[1].set_title("Netzwerkdichte")
axes[1].set_xlabel("Netzwerkdichte")
fig.supylabel("Textumfang")
fig.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system), "figures", "Umfang_auf_Zentralisierung_und_Netzwerkdichte.svg"))
plt.show()





plt.scatter(df["dep_var"], df["Netzwerkdichte"])
plt.title("Netzwerkdichte: Überschneidung zwischen beiden Verfahren")
plt.show()

print("Covariance. ", cov(1-df["dep_var"], df["token_count"]))

corr, _ = pearsonr(1- df["dep_var"], df["token_count"])
print('Pearsons correlation: %.3f' % corr)

corr, _ = spearmanr(1- df["dep_var"], df["token_count"])
print('Spearman correlation: %.3f' % corr)

df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1770, end_year=1970, epoch_length=20,
                      new_periods_column_name="periods")


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "MLP",
                                    "R": "R", "M": "M", "XE": "MLP"}}
df = full_genre_labels(df, replace_dict=replace_dict)