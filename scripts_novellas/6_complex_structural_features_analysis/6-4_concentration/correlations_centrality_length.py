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

medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"


infile_name = os.path.join(global_corpus_representation_directory(system), "SNA_novellas.csv")
print(infile_name)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)
matrix_obj = matrix_obj.add_metadata([genre_cat, year_cat, medium_cat, "Nachname", "Titel"])


cat_labels = ["N", "E", "0E", "XE", "R"]
matrix_obj = matrix_obj.reduce_to_categories(genre_cat, cat_labels)

matrix_obj = matrix_obj.eliminate(["Figuren"])

df = matrix_obj.data_matrix_df

plt.scatter(1-df['Netzwerkdichte'], df['Textumfang'])
plt.show()


print("Covariance. ", cov(1-df["Netzwerkdichte"], df["Textumfang"]))
corr, _ = pearsonr(1- df["Netzwerkdichte"], df["Textumfang"])
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(1- df["Netzwerkdichte"], df["Textumfang"])
print('Spearman correlation: %.3f' % corr)

df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1790, end_year=1880, epoch_length=20,
                      new_periods_column_name="periods")


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "N", "E": "E", "0E": "MLP",
                                    "R": "R", "M": "M", "XE": "MLP"}}
df = full_genre_labels(df, replace_dict=replace_dict)




#df = df[df.isin({medium_cat:["Familienblatt", "Anthologie", "Taschenbuch", "Rundschau"]}).any(1)]
df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)

#scaler = MinMaxScaler()
#df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

df_E = df[df["Gattungslabel_ED_normalisiert"] == "E"]
plt.scatter(1-df_E['Netzwerkdichte'], df_E['Textumfang'])
plt.title("Korrelation zwischen 1-Dichte und Länge für Erzählungen")
plt.show()


print("Covariance. ", cov(1-df_E["Netzwerkdichte"], df_E["Textumfang"]))
corr, _ = pearsonr(1- df_E["Netzwerkdichte"], df_E["Textumfang"])
print('Pearsons correlation Erzählungen: %.3f' % corr)
corr, _ = spearmanr(1- df_E["Netzwerkdichte"], df_E["Textumfang"])
print('Spearman correlation Erzählungen: %.3f' % corr)


df_E = df[df["Gattungslabel_ED_normalisiert"] == "N"]
plt.scatter(1-df_E['Netzwerkdichte'], df_E['Textumfang'])
plt.title("Korrelation zwischen 1-Dichte und Länge für Novellen")
plt.show()

print("Covariance. ", cov(1-df_E["Netzwerkdichte"], df_E["Textumfang"]))
corr, _ = pearsonr(1- df_E["Netzwerkdichte"], df_E["Textumfang"])
print('Pearsons correlation Novellen: %.3f' % corr)
corr, _ = spearmanr(1- df_E["Netzwerkdichte"], df_E["Textumfang"])
print('Spearman correlation Novellen: %.3f' % corr)


df_E = df[df["Gattungslabel_ED_normalisiert"] == "R"]
plt.scatter(1-df_E['Netzwerkdichte'], df_E['Textumfang'])
plt.title("Korrelation zwischen 1-Dichte und Länge für Romane")
plt.show()

print("Covariance. ", cov(1-df_E["Netzwerkdichte"], df_E["Textumfang"]))
corr, _ = pearsonr(1- df_E["Netzwerkdichte"], df_E["Textumfang"])
print('Pearsons correlation R: %.3f' % corr)
corr, _ = spearmanr(1- df_E["Netzwerkdichte"], df_E["Textumfang"])
print('Spearman correlation R: %.3f' % corr)

df_E = df[df["Gattungslabel_ED_normalisiert"] == "MLP"]
plt.scatter(1-df_E['Netzwerkdichte'], df_E['Textumfang'])
plt.title("Korrelation zwischen 1-Dichte und Länge für MLP")
plt.show()

print("Covariance. ", cov(1-df_E["Netzwerkdichte"], df_E["Textumfang"]))
corr, _ = pearsonr(1- df_E["Netzwerkdichte"], df_E["Textumfang"])
print('Pearsons correlation MLP: %.3f' % corr)
corr, _ = spearmanr(1- df_E["Netzwerkdichte"], df_E["Textumfang"])
print('Spearman correlation MLP: %.3f' % corr)

df.rename(columns={"Netzwerkdichte": "dep_var"}, inplace=True)

dep_var = "dep_var" # "Netzwerkdichte"

data = df.loc[:, (dep_var,genre_cat,medium_cat, "periods")]


