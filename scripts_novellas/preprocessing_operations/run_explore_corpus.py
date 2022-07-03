import os
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory, load_stoplist, vocab_lists_dicts_directory
from preprocessing.metadata_transformation import full_genre_labels, years_to_periods
from preprocessing.corpus import DocFeatureMatrix
import pandas as pd

system = "my_xps" #"wcph113"



infile_name = os.path.join(global_corpus_raw_dtm_directory(system), "DTM_lemmatized_l2_5000mfw.csv") # hier die Eingangsdatei der Featurerepräsentation wählen
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "my_colors.txt"))

rel_metadata = ["Gattungslabel_ED_normalisiert", "Gender", "Jahr_ED", "Medientyp_ED", "Kanon_Status"]

df_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)
df_obj = df_obj.add_metadata(rel_metadata)
new_df_obj = df_obj.reduce_to(rel_metadata, return_eliminated_terms_list=False)
# new_df_obj = new_df_obj.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=["N", "E", "0E", "XE", "M", "R"])

df = new_df_obj.data_matrix_df

df = years_to_periods(input_df=df, category_name="Jahr_ED",
                                          start_year=1790, end_year=1951, epoch_length=30,
                                          new_periods_column_name="periods30a")
print(df)

replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "sonstige Prosaerzählung", "0R" : "sonstige Journalinhalte", "0PE" : "sonstige Prosaerzählung", "0X_Essay" : "sonstige Journalinhalte", "0PB" : "sonstige Journalinhalte", "Lyrik" : "sonstige Journalinhalte", "Drama" : "sonstige Journalinhalte",
                                    "R": "Roman", "M": "Märchen", "XE": "sonstige Prosaerzählung", "V": "sonstige Journalinhalte", "V ": "sonstige Journalinhalte", "0P": "sonstige Journalinhalte", "0": "sonstige Journalinhalte"}}


df = full_genre_labels(df, replace_dict=replace_dict)


replace_dict = {"Medientyp_ED": {"Zeitschrift": "Journal", "Zeitung": "Journal", "Kalender": "Journal", "Rundschau" : "Journal", "Zyklus" : "Anthologie", "Roman" : "Werke", "(unbekannt)" : "selbst. Roman Buchdrucke",
                                    "Illustrierte": "Journal", "Sammlung": "Anthologie", "Nachlass": "Werke", "Jahrbuch":"Taschenbuch", "Monographie": "Werke"}}


df = full_genre_labels(df, replace_dict=replace_dict)

print("Einfache Korpusexploration:")
print(df.count())
rel_metadata = ["Gattungslabel_ED_normalisiert", "Gender", "Medientyp_ED", "Kanon_Status", "periods30a"]

for metadata in rel_metadata:
    column0 = str("Verteilung " + metadata)
    column1 = str("Verteilung "+ metadata + " (relativer Anteil)")
    new_df = pd.concat([df[metadata].value_counts(), df[metadata].value_counts(normalize=True)], axis=1)
    new_df.columns = [column0, column1]
    print(new_df)


grouped = df.groupby("periods30a")

for grouper, df in grouped:
    print('grouper:', grouper)
    for metadata in rel_metadata:
        column0 = str("Verteilung " + metadata)
        column1 = str("Verteilung " + metadata + " (relativer Anteil)")
        new_df = pd.concat([df[metadata].value_counts(), df[metadata].value_counts(normalize=True)], axis=1)
        new_df.columns = [column0, column1]
        print(new_df)

