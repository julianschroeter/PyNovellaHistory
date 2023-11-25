from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods

from preprocessing.sampling import equal_sample
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import os
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

system =  "my_xps" #  "wcph113" # "my_mac" # "wcph104"

medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"

data_matrix_filepath = os.path.join(global_corpus_representation_directory(system), "rom_top_matrix_comb.csv")

length_matrix_filepath = infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
matrix = DocFeatureMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = length_matrix_filepath, mallet=False)

matrix = matrix.reduce_to(["rom_top"])
matrix = matrix.add_metadata(["token_count"])

infile_path = os.path.join(local_temp_directory(system), "MaxDangerFearCharacters_novellas.csv")
matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix.data_matrix_df, metadata_csv_filepath=infile_path)
matrix = matrix.add_metadata(["max_value", "Angstempfinden"])

infile_path = os.path.join(local_temp_directory(system), "conll_based_networkdata-matrix-novellas.csv")
matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix.data_matrix_df, metadata_csv_filepath=infile_path)
matrix = matrix.add_metadata(["density", "centralization"])

infile_path = os.path.join(global_corpus_representation_directory(system), "speech_rep_Matrix.csv")
matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix.data_matrix_df, metadata_csv_filepath=infile_path)
matrix = matrix.add_metadata(["fraction_dirspeech","fraction_indirspeech","fraction_fid"])

df = matrix.data_matrix_df
df_with_NA = df.copy()
print(df_with_NA.loc["00769-00"])
df = df.dropna()

scaler = MinMaxScaler() # StandardScaler()
scaled_df = scaler.fit_transform(df.to_numpy())
df = pd.DataFrame(scaled_df, columns = ["rom_setting","length","danger","fear","density","centralization",
                                        "fraction_dirspeech","fraction_indirspeech" ,"fraction_fid"], index=df.index)
print(df)

df.to_csv(os.path.join(global_corpus_representation_directory(system), "doc_complex_features_matrix.csv"))