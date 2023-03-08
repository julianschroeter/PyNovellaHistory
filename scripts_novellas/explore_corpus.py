system = "wcph113" #"my_mac" "my_xps"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import pandas as pd
import os
import matplotlib.pyplot as plt

from preprocessing.corpus import DocFeatureMatrix
from preprocessing.presetting import global_corpus_representation_directory, vocab_lists_dicts_directory, load_stoplist, global_corpus_raw_dtm_directory
from preprocessing.metadata_transformation import full_genre_labels, years_to_periods


# infile_name = os.path.join(global_corpus_representation_directory(system), "DocThemesMatrix.csv")
infile_name = os.path.join(global_corpus_raw_dtm_directory(system), "raw_dtm_l1_lemmatized_use_idf_False2500mfw.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "my_colors.txt"))

dtm_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)

project_path = '/mnt/data/users/schroeter/PyNovellaHistory'

dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Nachname", "Titel", "Medientyp_ED", "Jahr_ED"])

cat_labels = ["N", "E", "0E", "XE", "M", "R"]
dtm_obj = dtm_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)


df = dtm_obj.data_matrix_df

print(df)



df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1790, end_year=2000, epoch_length=30,
                      new_periods_column_name="periods30a")



replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "sonstige Prosaerzählung",
                                    "R": "Roman", "M": "Märchen", "XE": "sonstige Prosaerzählung"}}
df = full_genre_labels(df, replace_dict=replace_dict)


replace_dict = {"Medientyp_ED": {"Buch": "Werke", "Jahrbuch": "Taschenbuch", "Roman": "Buch",
                                    "Werke": "Buch", "Zyklus": "Anthologie",
                                 "Monographie": "Buch", "Zeitschrift": "Zeitung",
                                 "Nachlass":"Buch", "Familienblatt": "Familienblatt/Illustrierte",
                                 "Illustrierte": "Familienblatt/Illustrierte"}}

# fpr convenenince different media types are normalized to a smaller set of types
replace_dict = {"Medientyp_ED": {"Zeitschrift": "Journal", "Zeitung": "Journal", "Kalender": "Journal",
                                 "Rundschau" : "Rundschau",
                                 "Zyklus" : "Anthologie", "Roman" : "Werke", "(unbekannt)" : "unbekannt",
                                    "Illustrierte": "Journal", "Sammlung": "Anthologie",
                                 "Nachlass": "Werke", "Jahrbuch":"Taschenbuch", "Monographie": "Werke"}}


df = full_genre_labels(df, replace_dict=replace_dict)

corpus_statistics = df.groupby(["periods30a", "Gattungslabel_ED_normalisiert"]).size()
df_corpus_statistics = pd.DataFrame(corpus_statistics)
df_corpus_statistics.unstack().plot(kind='bar', stacked=False, title= "Corpus Size for Genres and Periods",
                                    color=["green", "orange", "red", "blue", "grey"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(project_path, "figures", "Corpus_Size_for_Genres_and_Periods.png"))
plt.show()


corpus_statistics = df.groupby(["periods30a", "Medientyp_ED"]).size()
df_corpus_statistics = pd.DataFrame(corpus_statistics)
df_corpus_statistics.unstack().plot(kind='bar', stacked=False, title= "Corpus Size for Media Types and Periods",
                                    color=["green", "orange", "red", "blue", "grey", "cyan", "pink", "magenta"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(project_path, "figures", "Corpus_Size_for_Media_types_and_Periods.png"))
plt.show()

grouped = df.groupby(["periods30a", "Gattungslabel_ED_normalisiert"]).median()
df_grouped = pd.DataFrame(grouped)
print(df_grouped)
df_grouped = df_grouped.drop(columns=["Jahr_ED"])
