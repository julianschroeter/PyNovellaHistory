system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# standard libraries
import os

import statsmodels.api as sm
from statsmodels.formula.api import ols

from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import full_genre_labels, years_to_periods
# from my own modules:
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory

medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"


old_infile_name = os.path.join(global_corpus_representation_directory(system), "SNA_novellas.csv")
infile_name = os.path.join(global_corpus_representation_directory(system), "Network_Matrix_all.csv")
print(infile_name)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)
#matrix_obj = matrix_obj.add_metadata([genre_cat, year_cat, medium_cat, "Nachname", "Titel"])

df1 = matrix_obj.data_matrix_df

matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix_obj.data_matrix_df, metadata_csv_filepath=old_infile_name)
matrix_obj = matrix_obj.add_metadata(["Netzwerkdichte"])

length_infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=matrix_obj.data_matrix_df,
                              metadata_csv_filepath=length_infile_df_path)
matrix_obj = matrix_obj.add_metadata(["token_count"])

cat_labels = ["N", "E", "0E", "XE"]
cat_labels = ["N", "E"]
matrix_obj = matrix_obj.reduce_to_categories(genre_cat, cat_labels)

matrix_obj = matrix_obj.eliminate(["Figuren"])

df = matrix_obj.data_matrix_df

df.rename(columns={"scaled_centralization_conll": "dep_var"}, inplace=True) # "Netzwerkdichte"

df = df[~df["token_count"].isna()]


dep_var = "dep_var"
df.rename(columns={"Zentralisierung": "dep_var",
                   "Netzwerkdichte_conll":"Netzwerkdichte"}, inplace=True) # "Netzwerkdichte"
df["dep_var"] = df["dep_var"].astype(float)


df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1770, end_year=1970, epoch_length=20,
                      new_periods_column_name="periods")


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "MLP",
                                    "R": "R", "M": "M", "XE": "MLP"}}
df = full_genre_labels(df, replace_dict=replace_dict)




#df = df[df.isin({medium_cat:["Familienblatt", "Anthologie", "Taschenbuch", "Rundschau"]}).any(1)]
df = df[df.isin({medium_cat:["Familienblatt","Rundschau", "Anthologie", "Taschenbuch", "Buch", "Illustrierte", "Kalender", "Nachlass", "Sammlung", "Werke"
                             , "Zeitschrift", "Zeitung", "Zyklus"]}).any(axis=1)]

replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal", "Illustrierte": "Journal",
                                 "Werke": "Buch", "Nachlass": "Buch", "Kalender": "Taschenbuch",
                                 "Zyklus": "Anthologie", "Sammlung": "Anthologie"}}
df = full_genre_labels(df, replace_dict=replace_dict)


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

