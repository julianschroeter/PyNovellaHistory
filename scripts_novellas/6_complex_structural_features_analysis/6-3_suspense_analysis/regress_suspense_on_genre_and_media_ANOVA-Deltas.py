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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pymc as pm
from pymc import HalfCauchy, Model, Normal
import arviz as az
import bambi as bmb

media_name_cat = "Medium_ED"
medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"

scaler = StandardScaler()
scaler = MinMaxScaler()

columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"embedding_Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe", "Brustwarzen": "Erotik"}


dangers_list = ["Gewaltverbrechen", "Kampf", "Krieg", "Sturm", "Feuer", "Entführung"]
dangers_colors = ["cyan", "orange", "magenta", "blue", "pink", "purple"]
dangers_dict = dict(zip(dangers_list, dangers_colors[:len(dangers_list)]))


infile_path = os.path.join(local_temp_directory(system), "MaxDangerFearCharacters_novellas.csv") # "All_Chunks_Danger_FearCharacters_novellas.csv" # all chunks

matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)
df = matrix.data_matrix_df
df = df.rename(columns=columns_transl_dict)
df["doc_id"] = df.index
metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv" )
metadata_df = pd.read_csv(metadata_filepath, index_col=0)

matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=df, metadata_csv_filepath=metadata_filepath)
matrix = matrix.add_metadata([medium_cat, genre_cat, year_cat, "Kanon_Status"])
df = matrix.data_matrix_df
genre_labels = ["N", "E", "0E", "XE"]
genre_labels = ["N", "E"]
df = df[df.isin({genre_cat:genre_labels}).any(axis=1)]

replace_dict = {genre_cat: {"N": "N", "E": "E", "0E": "MLP",
                                    "R": "R", "M": "M", "XE": "MLP"}}
df = full_genre_labels(df, replace_dict=replace_dict)

df.rename(columns={"max_value": "dep_var"}, inplace=True) # "Netzwerkdichte"

dep_var = "dep_var"


df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1790, end_year=1950, epoch_length=20,
                      new_periods_column_name="periods")


df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)

#df = df[df[year_cat] >= 1850]

data = df.loc[:, (dep_var,genre_cat,medium_cat, "periods", "Kanon_Status")]





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
