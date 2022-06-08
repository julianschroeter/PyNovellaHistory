import pandas as pd

system = "my_xps" # "my_mac" # wcph104" #"wcph113" #

if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')


from preprocessing.presetting import global_corpus_representation_directory, language_model_path
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import years_to_periods, full_genre_labels

import matplotlib.pyplot as plt
import os
from scipy import stats


my_model_de = language_model_path(system)

infile_df_path = os.path.join(global_corpus_representation_directory(system), "Networkdata_document_Matrix_test.csv")
metadata_df_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_df_path, metadata_csv_filepath=metadata_df_path)
matrix_obj = matrix_obj.reduce_to(["Länge in Token"])
matrix_obj = matrix_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED", "Medientyp_ED"])

matrix_obj.data_matrix_df = years_to_periods(input_df=matrix_obj.data_matrix_df , category_name="Jahr_ED", start_year=1790, end_year=1950, epoch_length=20, new_periods_column_name="periods5a")


cat_labels = ["N", "E", "0E", "XE"]
matrix_obj = matrix_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)

df = matrix_obj.data_matrix_df

replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "other",
                                    "R": "novel", "M": "tale", "XE": "other"}}
df = full_genre_labels(df, replace_dict=replace_dict)

replace_dict = {"Medientyp_ED": {"Zeitschrift": "Journal", "Zeitung": "Journal",
                                 "Kalender": "Journal", "Rundschau" : "Journal",
                                 "Zyklus" : "Anthologie", "Roman" : "Werke",
                                 "(unbekannt)" : "Werke",
                                    "Illustrierte": "Journal", "Sammlung": "Anthologie",
                                 "Nachlass": "Werke", "Jahrbuch":"Taschenbuch",
                                 "Monographie": "Werke"}}
df = full_genre_labels(df, replace_dict=replace_dict)





print(df["Medientyp_ED"])



genre_grouped = df.groupby("Gattungslabel_ED_normalisiert")


# ANOVA with scipy.stats: calculte F-statistic and p-value
N_df = df['Länge in Token'][df['Gattungslabel_ED_normalisiert'] == 'Novelle']
E_df = df['Länge in Token'][df['Gattungslabel_ED_normalisiert'] == 'Erzählung']
otherE_df = df['Länge in Token'][df['Gattungslabel_ED_normalisiert'] == 'other']

F, p = stats.f_oneway(N_df, E_df)
print("F, p statistics of ANOVA test:", F, p)

grouped = df.groupby(["periods5a", "Gattungslabel_ED_normalisiert"]).mean()
grouped_df = pd.DataFrame(grouped)
grouped_df = grouped_df.drop(columns=["Jahr_ED"])
print(grouped_df)

grouped_df.unstack().plot(kind='bar', stacked=False,
                                       title=str("Length development for genres"),
                                       xlabel="Zeitverlauf von 1770 bis 1950", ylabel=str("length in tokens"))
plt.show()

df.boxplot(by=["Gattungslabel_ED_normalisiert"])
df.boxplot(by=["Medientyp_ED"])
plt.show()


grouped_df.unstack().plot(kind="line", stacked=False)
plt.show()

grouped = df.groupby(["periods5a", "Medientyp_ED"]).median()
grouped_df = pd.DataFrame(grouped)
grouped_df = grouped_df.drop(columns=["Jahr_ED"])
print(grouped_df)

grouped_df.unstack().plot(kind='bar', stacked=False,
                                       title=str("Length development for genres"),
                                       xlabel="Zeitverlauf von 1770 bis 1950", ylabel=str("length in tokens"))
plt.show()

grouped_df.boxplot(by=["Medientyp_ED"])
plt.show()


grouped_df.unstack().plot(kind="line", stacked=False)
plt.show()