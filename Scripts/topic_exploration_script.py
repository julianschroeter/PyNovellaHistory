from TopicModeling.Postprocessing import DocTopicMatrix
from Preprocessing.Presetting import global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory
from ClusteringNLP.myPCA import PC_df
from Preprocessing.MetadataTransformation import years_to_periods
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

system_name = "my_mac" # "wcph104"
data_matrix_filepath = os.path.join(mallet_directory(system_name), "doc-topics.txt")
language_model = os.path.join(set_DistReading_directory(system_name=system_name), "language_models", "model_de_lg")
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system_name), "Bibliographie.csv")
print(data_matrix_filepath)
print(metadata_csv_filepath)
df = pd.read_csv(os.path.join(global_corpus_representation_directory(system_name), "Bibliographie.csv"), index_col=0)

colors_list = load_stoplist(os.path.join(global_corpus_representation_directory(system_name), "my_colors.txt"))
textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system_name), "textanalytic_metadata.csv")
char_netw_infile_path = os.path.join(global_corpus_representation_directory(system_name), "Networkdata_document_Matrix.csv")

topic_doc_matrix = DocTopicMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = char_netw_infile_path, mallet=True)




topic_doc_matrix = topic_doc_matrix.adjust_doc_chunk_multiindex()
print(topic_doc_matrix.data_matrix_df)
topic_doc_matrix.data_matrix_df.to_csv(os.path.join(global_corpus_representation_directory(system_name), "test-doc-topics.csv"))

matrix = topic_doc_matrix.last_chunk()



#tod und liebe topics, manuell selegiert
matrix = matrix.reduce_to([66, 74, 76]) # = tragische Topics
matrix.data_matrix_df["topic_mean"] = matrix.data_matrix_df.mean(axis=1).values

matrix = matrix.add_metadata("Länge")
#matrix = matrix.add_metadata("Netzwerkdichte")
#matrix = matrix.add_metadata("Anteil")


matrix.data_matrix_df["log_Länge"] = matrix.data_matrix_df["Länge"].apply(lambda x: np.log(x))

matrix = matrix.reduce_to(["log_Länge", "topic_mean"])

matrix.data_matrix_df = matrix.data_matrix_df[["log_Länge", "topic_mean"]]

print(matrix.data_matrix_df)

matrix = DocTopicMatrix(data_matrix_df = matrix.data_matrix_df,
                        data_matrix_filepath=None, metadata_df=None, mallet=False,
                        metadata_csv_filepath=metadata_csv_filepath)


matrix = matrix.add_metadata("Gattungslabel_ED")
matrix = matrix.add_metadata("Jahr_ED")



matrix.data_matrix_df = years_to_periods(input_df=matrix.data_matrix_df, category_name="Jahr_ED",
                                          start_year=1750, end_year=1950, epoch_length=50,
                                          new_periods_column_name="periods")


matrix = matrix.reduce_to_categories("Gattungslabel_ED", ["E", "0E", "R", "M", "N", "Dorfgeschichte"])


matrix.data_matrix_df.replace({"Gattungslabel_ED": {"N": "Prosa",
                                                                                   "E": "Prosa",
                                                                                   "0E": "Prosa",
                                                                                   "R": "Roman",
                                                                                   "M": "Märchen"
                                                                                   }}, inplace=True)

matrix_red_for_val = matrix.eliminate(["periods", "Jahr_ED", "Gattungslabel_ED"])
matrix_genre = matrix.eliminate(["periods", "Jahr_ED"])
matrix_period = matrix.eliminate(["Jahr_ED", "Gattungslabel_ED"])

df = matrix_genre.data_matrix_df
print(df)
def scatter(df, colors_list):
    list_targetlabels = ", ".join(map(str, set(df.iloc[:,-1].values))).split(", ")
    zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))
    list_target = list(df.iloc[:,-1].values)
    colors_str = ", ".join(map(str, df.iloc[:,-1].values))
    colors_str = colors_str.translate(zipped_dict)
    list_targetcolors = [zipped_dict[label] for label in list_target]
    plt.figure(figsize=(10, 10))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=list_targetcolors, cmap='rainbow')
    plt.xlabel('Erste Komponente')
    plt.ylabel('Zweite Komponente')
    #plt.xlim(0,100)
    #plt.ylim(0, 1)
    mpatches_list = []
    for key, value in zipped_dict.items():
        patch = mpatches.Patch(color=value, label=key)
        mpatches_list.append(patch)
    plt.legend(handles=mpatches_list)
    plt.show()

print(df.sort_values(by=["topic_mean", "log_Länge"], ascending=False))
scatter(df, colors_list)

