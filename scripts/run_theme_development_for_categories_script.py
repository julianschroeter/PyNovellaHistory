
system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import pandas as pd
import os
import matplotlib.pyplot as plt

from preprocessing.corpus import DocFeatureMatrix
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory, load_stoplist
from preprocessing.metadata_transformation import full_genre_labels, years_to_periods


infile_name = os.path.join(global_corpus_representation_directory(system), "DocThemesMatrix.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
colors_list = load_stoplist(os.path.join(global_corpus_representation_directory(system), "my_colors.txt"))

dtm_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)
dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Nachname", "Titel", "Medium_ED", "Jahr_ED"])

cat_labels = ["N", "E", "0E", "XE", "M"]
dtm_obj = dtm_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)


df = dtm_obj.data_matrix_df

df = years_to_periods(input_df=df, category_name="Jahr_ED",start_year=1770, end_year=2000, epoch_length=20,
                      new_periods_column_name="periods20a")



replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "sonstige Prosaerzählung",
                                    "R": "Roman", "M": "Märchen", "XE": "sonstige Prosaerzählung"}}
df = full_genre_labels(df, replace_dict=replace_dict)


corpus_statistics = df.groupby(["periods20a", "Gattungslabel_ED_normalisiert"]).size()
df_corpus_statistics = pd.DataFrame(corpus_statistics)
df_corpus_statistics.unstack().plot(kind='bar', stacked=False, title= "Größe des Korpus")
plt.show()

grouped = df.groupby(["periods20a", "Gattungslabel_ED_normalisiert"]).mean()
df_grouped = pd.DataFrame(grouped)
print(df_grouped)
df_grouped = df_grouped.drop(columns=["Jahr_ED"])

for column_name in df_grouped.columns.values.tolist():

    df_grouped[column_name].unstack().plot(kind='bar', stacked=False,
                                       title=str("Entwicklung des Bezugs zu "+str(column_name)),
                                       xlabel="Zeitverlauf von 1770 bis 1950", ylabel=str("Anteil von Wörtern, die indizieren:"+str(column_name)))
    plt.show()
    df_grouped[column_name].unstack().plot(kind='line', stacked=False, title=str("Entwicklung des Bezugs zu "+str(column_name)),
                                       ylabel=str("Anteil von Wörtern, die indizieren:"+str(column_name)))
    plt.show()




df_N = df[df["Gattungslabel_ED_normalisiert"] == "N"]
df_N_periods_grouped = df_N.groupby(["periods20a"]).mean()
df_N_periods_grouped["Marseille"].plot(kind="bar", title="Anteil frz/span/ital Novellen")
plt.show()

df_E = df[df["Gattungslabel_ED_normalisiert"] == "E"]
df_E_periods_grouped = df_E.groupby(["periods20a"]).mean()
df_E_periods_grouped["Marseille"].plot(kind="bar", title="Anteil frz/span/ital Erzählungen")
plt.show()

df_0E = df[df["Gattungslabel_ED_normalisiert"] == "0E"]
df_0E_periods_grouped = df_0E.groupby(["periods20a"]).mean()
df_0E_periods_grouped["Marseille"].plot(kind="bar",  title="Anteil frz/span/ital sonstige Erzählprosa")
plt.show()

df_R = df[df["Gattungslabel_ED_normalisiert"] == "R"]
df_R_periods_grouped = df_R.groupby(["periods20a"]).mean()
df_R_periods_grouped["Marseille"].plot(kind="bar",
                                        title="hist. Entwicklung: frz/span/ital Wörter in Romanen")
plt.show()

df_M = df[df["Gattungslabel_ED_normalisiert"] == "M"]
df_M_periods_grouped = df_M.groupby(["periods20a"]).mean()

df_M_periods_grouped["Marseille"].plot(kind="bar",
                                        title="hist. Entwicklung: frz/span/ital Wörter in Märchen")
plt.show()