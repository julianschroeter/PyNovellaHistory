from TopicModeling.Postprocessing import DocTopicMatrix
from preprocessing.presetting import global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory
from clustering.my_pca import PC_df
from preprocessing.metadata_transformation import years_to_periods
import os
import pandas as pd
import numpy as np

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

matrix = matrix.eliminate(["Länge"])

matrix = DocTopicMatrix(data_matrix_df = matrix.data_matrix_df,
                        data_matrix_filepath=None, metadata_df=None, mallet=False,
                        metadata_csv_filepath=metadata_csv_filepath)


matrix = matrix.add_metadata("Gattungslabel_ED")
matrix = matrix.add_metadata("Jahr_ED")



matrix.data_matrix_df = years_to_periods(input_df=matrix.data_matrix_df, category_name="Jahr_ED",
                                          start_year=1750, end_year=1950, epoch_length=50,
                                          new_periods_column_name="periods")


matrix = matrix.reduce_to_categories("Gattungslabel_ED", ["E", "0E", "R", "M", "N", "Dorfgeschichte"])


matrix.data_matrix_df.replace({"Gattungslabel_ED": {"N": "Novelle",
                                                                                   "E": "Erzählung",
                                                                                   "0E": "sonstige Prosa",
                                                                                   "R": "Roman",
                                                                                   "M": "Märchen"
                                                                                   }}, inplace=True)

matrix_red_for_val = matrix.eliminate(["periods", "Jahr_ED", "Gattungslabel_ED"])
matrix_genre = matrix.eliminate(["periods", "Jahr_ED"])
matrix_period = matrix.eliminate(["Jahr_ED", "Gattungslabel_ED"])

matrix_val = DocTopicMatrix(data_matrix_df = matrix_red_for_val.data_matrix_df,
                            data_matrix_filepath=None, metadata_df=None, mallet=False,
                            metadata_csv_filepath=textanalytic_metadata_filepath)
matrix_val = matrix_val.add_metadata("Ende")

matrix_val.data_matrix_df["Ende"] = matrix_val.data_matrix_df["Ende"].fillna("other", inplace=False)
print(matrix_val.data_matrix_df)
matrix_val = matrix_val.reduce_to_categories("Ende", ["schauer", "tragisch", "Liebesglück", "nein", "Erkenntnis",
                                                      "Entsagung", "tragisch (schwach)", "unbestimmt", "other"])


matrix_val.data_matrix_df.replace({"Ende": {"tragisch": "tragisch",
                                                        "schauer": "tragisch",
                                                        "Liebesglück": "positiv",
                                                       "nein": "positiv",
                                                       "Erkenntnis": "positiv",
                                                        "tragisch (schwach)" : "tragisch",
                                                        "unbestimmt" : "positiv",
                                                        "Entsagung" : "positiv"

                                                       }}, inplace=True)



pc_df = PC_df(input_df=matrix_genre.data_matrix_df)
pc_df.generate_pc_df(n_components=2)

#pc_df.pc_target_df = pc_df.pc_target_df[pc_df.pc_target_df.isin({"target": ["tragisch", "positiv"] }).any(1)]


print(pc_df.pc_target_df)
print(pc_df.pc_target_df.sort_values(by=["PC_1", "PC_2"] ,axis=0))
print(pc_df.component_loading_df.loc["PC_1"].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.loc["PC_2"].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.loc["PC_1"].sort_values(ascending=True)[:20])
print(pc_df.component_loading_df.loc["PC_2"].sort_values(ascending=True)[:20])
print(pc_df.pca.explained_variance_)
pc_df.scatter(colors_list)
