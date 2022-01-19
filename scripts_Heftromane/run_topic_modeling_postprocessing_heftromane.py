from TopicModeling.Postprocessing import DocTopicMatrix
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, local_temp_directory, global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory
import os
import pandas as pd


system_name = "wcph113" # "my_mac" # "wcph104"
data_matrix_filepath = os.path.join(local_temp_directory(system_name), "heftr_doc-topics-30t.txt")
language_model = language_model_path(system_name)
metadata_csv_filepath = os.path.join(local_temp_directory(system_name=system_name), "Metadatentabelle_Heftromane.csv")

colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system_name), "my_colors.txt"))
topic_doc_matrix = DocTopicMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = metadata_csv_filepath, mallet=True)

df_meta = topic_doc_matrix.metadata_df
print(df_meta)


topic_doc_matrix = topic_doc_matrix.adjust_doc_chunk_multiindex()
topic_doc_matrix.data_matrix_df.to_csv(os.path.join(global_corpus_representation_directory(system_name), "heftromane_test-doc-topics.csv"))
df_all_segments = topic_doc_matrix.data_matrix_df
matrix = topic_doc_matrix.mean_doclevel()
doc_topic_df = matrix.data_matrix_df

matrix = matrix.reduce_to([22]) # = bestimmtes topic
df = matrix.data_matrix_df
print(df)

matrix = matrix.add_metadata("title")


df_title = matrix.data_matrix_df

print(df_title)