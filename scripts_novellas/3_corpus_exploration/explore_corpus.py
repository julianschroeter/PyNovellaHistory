system = "my_xps" #"wcph113" #"my_mac"
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
project_path = '/home/julian/Documents/CLS_temp'


dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Nachname", "Titel", "Medientyp_ED", "Jahr_ED"])

cat_labels = ["N", "E", "0E", "XE", "M", "R"]
dtm_obj = dtm_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)


df = dtm_obj.data_matrix_df

print(df)

period_var = "Perioden"

df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1760, end_year=2000, epoch_length=30,
                      new_periods_column_name=period_var)



replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "sonstige Prosaerzählung",
                                    "R": "Roman", "M": "Märchen", "XE": "sonstige Prosaerzählung"}}
df = full_genre_labels(df, replace_dict=replace_dict)


replace_dict = {"Medientyp_ED": {"Buch": "Werke", "Jahrbuch": "Taschenbuch", "Roman": "Buch",
                                    "Werke": "Buch", "Zyklus": "Anthologie",
                                 "Monographie": "Buch", "Zeitschrift": "Zeitung",
                                 "Nachlass":"Buch", "Familienblatt": "Familienblatt/Illustrierte",
                                 "Illustrierte": "Familienblatt/Illustrierte"}}

# for convenenince different media types are normalized to a smaller set of types
replace_dict = {"Medientyp_ED": {"Zeitschrift": "Journal", "Zeitung": "Journal", "Kalender": "Journal",
                                 "Rundschau" : "Rundschau", "Werke":"Buch",
                                 "Zyklus" : "Anthologie", "Roman" : "Buch", "(unbekannt)" : "verm. Buch",
                                    "Illustrierte": "Journal", "Sammlung": "Anthologie",
                                 "Nachlass": "Buch", "Jahrbuch":"Taschenbuch", "Monographie": "Buch"}}

df = full_genre_labels(df, replace_dict=replace_dict)


title_en = "Corpus Size for Genres and Periods"
title_de = "Zusammensetzung des Korpus nach Perioden und Genres"
corpus_statistics = df.groupby([period_var, "Gattungslabel_ED_normalisiert"]).size()
df_corpus_statistics_genre = pd.DataFrame(corpus_statistics)
df_corpus_statistics_genre.unstack().plot(kind='bar', stacked=False, title= title_de ,
                                    color=["green", "orange", "red", "blue", "grey"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(project_path, "figures", "Corpus_Size_for_Genres_and_Periods.svg"))
plt.show()


corpus_statistics = df.groupby([period_var, "Medientyp_ED"]).size()
df_corpus_statistics_media = pd.DataFrame(corpus_statistics)

title_en = "Corpus Size for Media and Periods"
title_de = "Zusammensetzung des Korpus nach Perioden und Medienformaten"

df_corpus_statistics_media.unstack().plot(kind='bar', stacked=False, title= title_de,
                                    color=["lightgreen", "orange", "purple", "lightblue", "grey", "cyan", "pink", "magenta"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(project_path, "figures", "Corpus_Size_for_Media_types_and_Periods.svg"))
plt.show()

corpus_statistics = df.groupby(["Medientyp_ED"]).size()
df_corpus_statistics_media_nontemporal = pd.DataFrame(corpus_statistics)
corpus_statistics = df.groupby(["Gattungslabel_ED_normalisiert"]).size()
df_corpus_statistics_genre_nontemporal = pd.DataFrame(corpus_statistics)
