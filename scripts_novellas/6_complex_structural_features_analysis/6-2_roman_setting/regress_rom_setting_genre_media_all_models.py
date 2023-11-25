import statistics

from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory
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

data_matrix_filepath = os.path.join(global_corpus_representation_directory(system), "toponym_share_Matrix.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

matrix = DocFeatureMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = metadata_filepath, mallet=False)

matrix = matrix.reduce_to(["rom_top"])
matrix = matrix.add_metadata([genre_cat, medium_cat, year_cat])

cat_labels = ["N", "E", "0E", "XE", "R"]
cat_labels = ["N", "E"]

matrix = matrix.reduce_to_categories(genre_cat, cat_labels)
replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "N", "E": "E", "0E": "MLP",
                                    "R": "R", "M": "M", "XE": "MLP"}}


df = matrix.data_matrix_df
df = full_genre_labels(df, replace_dict=replace_dict)

df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)

scaler = MinMaxScaler() # StandardScaler()

df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

print(df)

df.rename(columns={"rom_top": "dep_var"}, inplace=True) # "Netzwerkdichte"

#df = df[~df["token_count"].isna()]

dep_var = "dep_var"


df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1790, end_year=1950, epoch_length=20,
                      new_periods_column_name="periods")


#df = df[df.isin({medium_cat:["Familienblatt", "Anthologie", "Taschenbuch", "Rundschau"]}).any(1)]
df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)

#scaler = MinMaxScaler()
#df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

#df = df[df[year_cat] >= 1850]

data = df.loc[:, (dep_var,genre_cat,medium_cat, "periods")]




print("Model without interaction and only for media:")
lm = ols('dep_var ~ C(Medientyp_ED, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model including an interaction between genre and medium:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum) + C(periods, Sum) + C(Gattungslabel_ED_normalisiert, Sum)*C(Medientyp_ED, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model including an interaction between genre and periods:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum) + C(periods, Sum) + C(Gattungslabel_ED_normalisiert, Sum)*C(periods, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model including an interaction between medium and periods:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum) + C(periods, Sum) + C(Medientyp_ED, Sum)*C(periods, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model 1.c without interaction, including time and genre:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum) + C(periods, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model 1.b without interaction, including time and media:")
lm = ols('dep_var ~ C(Medientyp_ED, Sum) + C(periods, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model without interaction and only for time:")
lm = ols('dep_var ~ C(periods, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model without interaction and only for genre:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model without interaction and without period and genre:")
lm = ols('dep_var ~ C(Medientyp_ED, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model 2.a including an interaction between all variables:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum) + C(periods, Sum) + C(Medientyp_ED, Sum)*C(periods, Sum)*C(Gattungslabel_ED_normalisiert, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model 1.a without interaction:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum) + C(periods, Sum)',
                 data=data).fit()


print(lm.summary())

anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)
deltas_df = anova_table
deltas_df["adj_sum_sq"] = anova_table["sum_sq"] / anova_table["df"]
print(deltas_df)

print("Delta Genre: ", deltas_df.iloc[0,4] / (deltas_df.iloc[0,4] + deltas_df.iloc[1,4] + deltas_df.iloc[2,4]))
print("Delta Medium: ", deltas_df.iloc[1,4] / (deltas_df.iloc[0,4] + deltas_df.iloc[1,4] + deltas_df.iloc[2,4]))
print("Delta time: ", deltas_df.iloc[2,4] / (deltas_df.iloc[0,4] + deltas_df.iloc[1,4] + deltas_df.iloc[2,4]))

