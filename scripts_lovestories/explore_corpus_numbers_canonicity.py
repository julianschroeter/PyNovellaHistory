import os
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory, load_stoplist, vocab_lists_dicts_directory
from preprocessing.metadata_transformation import full_genre_labels
from preprocessing.corpus import DocFeatureMatrix
import pandas as pd

# if the functions for setting the filepaths from the preprocessing.presetting module are used, the name of the Computer can be specified in thy "system" variable.
system = "my_xps" # "wcph113" #

# Skript ist erst im Aufbau

infile_name = os.path.join(global_corpus_raw_dtm_directory(system), "raw_dtm_l1_lemmatized_use_idf_False2500mfw.csv") # set the filepath for the document-feature-matrix file (see data folder)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv") # set the filepath for the metadata.csv filepath (see data folder)

# metadata categories to be added
rel_metadata = ["Gattungslabel_ED_normalisiert", "Jahr_ED", "Medientyp_ED", "Gender", "Kanon_Status"]

# generate instance of a corpus object (see class DocFeatureMatrix in the preprocessing module)
df_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)
df_obj = df_obj.add_metadata(rel_metadata)
new_df_obj = df_obj.reduce_to(rel_metadata, return_eliminated_terms_list=False)

# data can be reduced to selected genres
# new_df_obj = new_df_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=["N", "E", "0E", "XE", "M", "R"])

df = new_df_obj.data_matrix_df

# for convenience, abbrieviations can be replaced by meaningful genre names
replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "sonstige Prosaerzählung", "0R" : "sonstige Journalinhalte", "0PE" : "sonstige Prosaerzählung", "0X_Essay" : "sonstige Journalinhalte", "0PB" : "sonstige Journalinhalte", "Lyrik" : "sonstige Journalinhalte", "Drama" : "sonstige Journalinhalte",
                                    "R": "Roman", "M": "Märchen", "XE": "sonstige Prosaerzählung", "V": "sonstige Journalinhalte", "V ": "sonstige Journalinhalte", "0P": "sonstige Journalinhalte", "0": "sonstige Journalinhalte"}}
df = full_genre_labels(df, replace_dict=replace_dict)

# fpr convenenince different media types are normalized to a smaller set of types
replace_dict = {"Medientyp_ED": {"Zeitschrift": "Journal", "Zeitung": "Journal", "Kalender": "Journal", "Rundschau" : "Journal", "Zyklus" : "Anthologie", "Roman" : "Werke", "(unbekannt)" : "selbst. Roman Buchdrucke",
                                    "Illustrierte": "Journal", "Sammlung": "Anthologie", "Nachlass": "Werke", "Jahrbuch":"Taschenbuch", "Monographie": "Werke"}}


df = full_genre_labels(df, replace_dict=replace_dict)

print(df.sort_index())

# print exploratory aspects of the corpus structure
print("first corpus exploration for all metadata categories:")
print(df.count())

for metadata in rel_metadata:
    column0 = str("distrubtion of " + metadata)
    column1 = str("distribution of "+ metadata + " (relative share)")
    new_df = pd.concat([df[metadata].value_counts(), df[metadata].value_counts(normalize=True)], axis=1)
    new_df.columns = [column0, column1]
    print(new_df)

df_female = df[df["Gender"] == "f"]
df_male = df[df["Gender"] == "m"]

print(df_female.shape[0])
print(df_female[df_female["Kanon_Status"] == 3].shape[0])
print(df_female[df_female["Kanon_Status"] == 2].shape[0])
print((df_female[df_female["Kanon_Status"] == 3].shape[0]) / df_female.shape[0])
print((df_female[df_female["Kanon_Status"] == 3].shape[0] + df_female[df_female["Kanon_Status"] == 2].shape[0]) / df_female.shape[0])
print((df_female[df_female["Kanon_Status"] == 0].shape[0]) / df_female.shape[0])
print((df_female[df_female["Kanon_Status"] == 1].shape[0]) / df_female.shape[0])
print((df_female[df_female["Kanon_Status"] == 2].shape[0]) / df_female.shape[0])

print(df_male.shape[0])
print(df_male[df_male["Kanon_Status"] == 3].shape[0])
print(df_male[df_male["Kanon_Status"] == 2].shape[0])
print((df_male[df_male["Kanon_Status"] == 3].shape[0] ) / df_male.shape[0])
print((df_male[df_male["Kanon_Status"] == 3].shape[0] + df_male[df_male["Kanon_Status"] == 2].shape[0]) / df_male.shape[0])
print((df_male[df_male["Kanon_Status"] == 0].shape[0] ) / df_male.shape[0])
print((df_male[df_male["Kanon_Status"] == 1].shape[0] ) / df_male.shape[0])
print((df_male[df_male["Kanon_Status"] == 2].shape[0] ) / df_male.shape[0])