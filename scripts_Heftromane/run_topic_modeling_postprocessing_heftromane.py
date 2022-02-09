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

print(topic_doc_matrix.data_matrix_df)



topic_doc_matrix.data_matrix_df = topic_doc_matrix.data_matrix_df.droplevel("chunk_count")

matrix = topic_doc_matrix
print(matrix.data_matrix_df)
# matrix = topic_doc_matrix.mean_doclevel()


# matrix = matrix.reduce_to([1]) # = bestimmtes topic


matrix = matrix.add_metadata("title")


df_title = matrix.data_matrix_df

print(df_title)

from clustering.my_pca import PC_df
from preprocessing.presetting import global_corpus_representation_directory, load_stoplist, global_corpus_raw_dtm_directory
import os


colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system_name), "my_colors.txt"))
print(colors_list)

pc_df = PC_df(input_df= df_title)

pc_df.generate_pc_df(n_components=2)


print(pc_df.pc_target_df.sort_values(by=["PC_2"], axis=0, ascending=False))
print(pc_df.component_loading_df.loc["PC_1"].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.loc["PC_2"].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.loc["PC_1"].sort_values(ascending=True)[:20])
print(pc_df.component_loading_df.loc["PC_2"].sort_values(ascending=True)[:20])
print(pc_df.pca.explained_variance_)
pc_df.scatter(colors_list)
