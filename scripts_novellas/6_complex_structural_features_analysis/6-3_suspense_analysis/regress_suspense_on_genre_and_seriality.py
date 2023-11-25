system = "my_xps" #"wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/PyNovellaHistory')

import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr, siegelslopes
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols


media_name_cat = "Medium_ED"
medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"


novellas_infilepath = os.path.join(local_temp_directory(system),  "AllChunksDangerFearCharacters_novellas_episodes_scaled.csv")


novellas_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")



novellas_df = pd.read_csv(novellas_infilepath, index_col = 0)
novellas_dtm_obj = DocFeatureMatrix(data_matrix_filepath=novellas_infilepath, metadata_csv_filepath=novellas_metadata_filepath)
novellas_dtm_obj = novellas_dtm_obj.add_metadata(["Titel", "Nachname","Jahr_ED","Gattungslabel_ED_normalisiert","Medium_ED", "Kanon_Status", "seriell"])
novellas_df = novellas_dtm_obj.data_matrix_df
novellas_df_full_media = standardize_meta_data_medium(df=novellas_df, medium_column_name="Medium_ED")

novellas_df = novellas_df_full_media.drop(columns=["Medium_ED", "medium"])


df = novellas_df

df = df[df["doc_chunk_id"].map(len) == 8]


labels_list = ["R", "M", "E", "N", "0E", "XE", "0P", "0PB", "krimi", "abenteuer", "krieg"]
labels_list = ["E", "N"]
df = df[df.isin({genre_cat: labels_list}).any(axis=1)]

other_cat_labels_list =  ["Taschenbuch", "Familienblatt", "Rundschau"]
other_cat_labels_list = ["Kleist", "Goethe", "Hoffmann" , "Eichendorff","Tieck", "Stifter", "Storm", "Keller", "Meyer", "Schnitzler", "Mann", "Musil"]
#whole_df = whole_df[whole_df.isin({"author": other_cat_labels_list}).any(1)]


replace_dict = {genre_cat: {"N": "MLP", "E": "MLP", "0E": "MLP", "XE": "MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen",
                          "krimi": "Spannungs-Heftroman", "abenteuer": "Spannungs-Heftroman", "krieg": "Spannungs-Heftroman"}}

replace_dict = {genre_cat: {"N": "Novelle", "E": "Erzählung", "0E": "sonst. MLP",
                                                  "0P": "non-fiction", "0PB":"non-fiction",
                                    "R": "Roman", "M": "Märchen", "XE": "sonst. MLP"}}


df = full_genre_labels(df, replace_dict=replace_dict)

replace_dict = {"seriell": {"True": "Serie", "TRUE": "Serie", "vermutlich": "Serie",
                                                  "False": "nicht-seriell", "FALSE":"nicht-seriell"}}

df = full_genre_labels(df, replace_dict=replace_dict)

#whole_df = df.drop(df['max_value'].idxmax())
whole_df = df.copy()
serial_status_list = ["Serie", "nicht-seriell"]
whole_df = whole_df[whole_df.isin({"seriell": serial_status_list}).any(axis=1)]

# coefficients for linear suspense model based on correlation in annotations: suspense = max_danger_level + 0.725 * Fear_level
whole_df["lin_susp_model"] = whole_df.apply(lambda x: x.max_value + (0.725 * x.Angstempfinden), axis=1)

scaler = MinMaxScaler()

scaled_values = scaler.fit_transform(whole_df[["Gewaltverbrechen", "Kampf", "Entführung", "Krieg", "Spuk","max_value", "Angstempfinden", "Sturm", "Feuer", "lin_susp_model"]])
whole_df[["Gewaltverbrechen", "Kampf", "Entführung", "Krieg","Spuk","max_value", "Angstempfinden", "Sturm", "Feuer", "lin_susp_model"]] = scaled_values
#whole_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

whole_df = years_to_periods(input_df=whole_df, category_name=year_cat
                            , start_year=1790,end_year=19300, epoch_length=20,
                            new_periods_column_name="periods")

#whole_df = whole_df[whole_df["lin_susp_model"] != 0]
df = whole_df.copy()

df.rename(columns={"max_value": "dep_var"}, inplace=True) # "Netzwerkdichte"

dep_var = "dep_var"
serial_cat = "seriell"

#df = df[df[year_cat] >= 1850]

data = df.loc[:, (dep_var,genre_cat,serial_cat, "periods", "Kanon_Status")]


print("Model including an interaction between all variables, only product:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum) * C(seriell, Sum) * C(periods, Sum)',
                 data=data).fit()

print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)
deltas_df = anova_table
deltas_df["adj_sum_sq"] = anova_table["sum_sq"] / anova_table["df"]
print(deltas_df)

print("Model 1.a without interaction:")
lm = ols('dep_var ~ C(seriell, Sum) + C(Gattungslabel_ED_normalisiert, Sum) + C(periods, Sum)',
                 data=data).fit()


print("Model without interaction and only for media:")
lm = ols('dep_var ~ C(seriell, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model only for Kanon_Status:")
lm = ols('dep_var ~ C(Kanon_Status, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model including an interaction between genre and medium:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)+C(seriell, Sum) + C(periods, Sum) + C(Gattungslabel_ED_normalisiert, Sum)*C(seriell, Sum)',
                 data=data).fit()
print("Model including an interaction between genre and periods:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)+C(seriell, Sum) + C(periods, Sum) + C(Gattungslabel_ED_normalisiert, Sum)*C(periods, Sum)',
                 data=data).fit()
print("Model including an interaction between medium and periods:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)+C(seriell, Sum) + C(periods, Sum) + C(seriell, Sum)*C(periods, Sum)',
                 data=data).fit()


print("Model 1.c without interaction, including time and genre:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum) + C(periods, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)
deltas_df = anova_table
deltas_df["adj_sum_sq"] = anova_table["sum_sq"] / anova_table["df"]
print(deltas_df)

print("Model 1.a without interaction, including time, media, genre, and Kanon_Status:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum) + C(seriell, Sum) +C(periods, Sum) + C(Kanon_Status, Sum)',
                 data=data).fit()
print(lm.summary())

anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)
deltas_df = anova_table
deltas_df["adj_sum_sq"] = anova_table["sum_sq"] / anova_table["df"]
print(deltas_df)

print("Delta Genre: ", deltas_df.iloc[0,4] / (deltas_df.iloc[0,4] + deltas_df.iloc[1,4] + deltas_df.iloc[2,4]+ deltas_df.iloc[3,4]))
print("Delta Medium: ", deltas_df.iloc[1,4] / (deltas_df.iloc[0,4] + deltas_df.iloc[1,4] + deltas_df.iloc[2,4]+ deltas_df.iloc[3,4]))
print("Delta time: ", deltas_df.iloc[2,4] / (deltas_df.iloc[0,4] + deltas_df.iloc[1,4] + deltas_df.iloc[2,4]+ deltas_df.iloc[3,4]))
print("Delta Kanonstatus: ", deltas_df.iloc[3,4] / (deltas_df.iloc[0,4] + deltas_df.iloc[1,4] + deltas_df.iloc[2,4]+ deltas_df.iloc[3,4]))

print("Model 2 with interaction only between time and genre:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum) * C(periods, Sum)',
                 data=data).fit()
print(lm.summary())

anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)
deltas_df = anova_table
deltas_df["adj_sum_sq"] = anova_table["sum_sq"] / anova_table["df"]
print(deltas_df)

print("Model 1.b without interaction, including time and media:")
lm = ols('dep_var ~ C(seriell, Sum) + C(periods, Sum)',
                 data=data).fit()
print("Model without interaction and only for time:")
lm = ols('dep_var ~ C(periods, Sum)',
                 data=data).fit()
print("Model without interaction and only for genre:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)',
                 data=data).fit()
print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

print("Model without interaction and without period and genre:")
lm = ols('dep_var ~ C(seriell, Sum)',
                 data=data).fit()


print("Model 2.a including an interaction between all variables:")
lm = ols('dep_var ~ C(seriell, Sum)*C(periods, Sum)*C(Gattungslabel_ED_normalisiert, Sum) * C(Kanon_Status, Sum)',
                 data=data).fit()

print(lm.summary())
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)
deltas_df = anova_table
deltas_df["adj_sum_sq"] = anova_table["sum_sq"] / anova_table["df"]
print(deltas_df)

print("Model 1.a without interaction:")
lm = ols('dep_var ~ C(Gattungslabel_ED_normalisiert, Sum)+C(seriell, Sum) + C(periods, Sum)',
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






#results_str = "\n".join([anova_ann, anova_table.to_string(), ])
#with open(os.path.join(local_temp_directory(system), "genre_media_regression_on_suspense_results.txt"), "w", encoding="utf8") as file:
 #   file.write(results_str)