system =  "my_xps"  # "my_mac"


import pandas as pd
import os
import matplotlib.pyplot as plt

from preprocessing.corpus import DocFeatureMatrix
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory, load_stoplist, vocab_lists_dicts_directory
from preprocessing.metadata_transformation import full_genre_labels, years_to_periods
import seaborn as sns

colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "my_colors.txt"))
infile_name = os.path.join(global_corpus_representation_directory(system), "SNA_novellas.csv")
print(infile_name)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


dtm_obj = DocFeatureMatrix(data_matrix_filepath=infile_name, metadata_csv_filepath= metadata_filepath)
dtm_obj = dtm_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED"])



cat_labels = ["N", "E", "0E", "XE", "R"]
dtm_obj = dtm_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)

dtm_obj = dtm_obj.eliminate(["Figuren"])

df = dtm_obj.data_matrix_df

df = years_to_periods(input_df=df, category_name="Jahr_ED", start_year=1790, end_year=1880, epoch_length=10,
                      new_periods_column_name="periods")

df = df[df["periods"] != 0]

df['density-centrality'] = df.apply(lambda x : x['Netzwerkdichte'] * x["Anteil Figuren mit degree centrality == 1"], axis=1)

replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "sonstige Prosaerzählung",
                                    "R": "Roman", "M": "Märchen", "XE": "sonstige Prosaerzählung"}}
df = full_genre_labels(df, replace_dict=replace_dict)


corpus_statistics = df.groupby(["periods", "Gattungslabel_ED_normalisiert"]).size()
df_corpus_statistics = pd.DataFrame(corpus_statistics)
df_corpus_statistics.unstack().plot(kind='bar', stacked=False, title= "Größe des Korpus")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="Netzwerkdichte", hue="Gattungslabel_ED_normalisiert",
             palette=["red","orange","green","blue"])
plt.title("Netzwerkdichte")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="Anteil Figuren mit degree centrality == 1", hue="Gattungslabel_ED_normalisiert",
             palette=["red","orange","green","blue"])
plt.title("Anteil der Figuren mit degree centrality == 1")
plt.show()

sns.lineplot(data=df, x="Jahr_ED", y="density-centrality", hue="Gattungslabel_ED_normalisiert")
plt.title("Produkt aus Dichte * Anteil voll verbundener Figuren")
plt.show()

grouped = df.groupby(["periods", "Gattungslabel_ED_normalisiert"]).mean()
df_grouped = pd.DataFrame(grouped)
print(df_grouped)
df_grouped = df_grouped.drop(columns=["Jahr_ED"])

for column_name in df_grouped.columns.values.tolist():

    df_grouped[column_name].unstack().plot(kind='bar', stacked=False,
                                       title=str("Entwicklung: "+str(column_name)),
                                       xlabel="Zeitverlauf von 1750 bis 1950", ylabel=str("Merkmal:"+str(column_name)))
    plt.show()
    df_grouped[column_name].unstack().plot(kind='line', stacked=False, title=str("Entwicklung: "+str(column_name)),
                                       ylabel=str("Merkmal:"+str(column_name)))
    plt.show()