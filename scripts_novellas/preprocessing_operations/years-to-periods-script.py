from preprocessing.corpus import DTM
from preprocessing.metadata_transformation import years_to_periods
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory
from preprocessing.presetting import load_stoplist
import os
import pandas as pd

system = "wcph113" # "my_mac" # "wcph104" "my_xps"

tdm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), "raw_dtm_5000mfw_lemmatized_tfidf_5000mfw.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

dtm_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), "dtm_lemma_tfidf5000semantic_periods100a.csv")
stopword_list_filepath = os.path.join(global_corpus_representation_directory(system), "stopwords_all.txt")
stopword_list = load_stoplist(stopword_list_filepath)

semanticword_list_filepath = os.path.join(global_corpus_representation_directory(system), "semantic_word_list_corpus_vocab.txt")
semanticword_list = load_stoplist(semanticword_list_filepath)

metadata_df = pd.read_csv(filepath_or_buffer=metadata_filepath, index_col=0)
print(metadata_df["Jahr_ED"])

metadata_df_periods = years_to_periods(input_df=metadata_df, category_name="Jahr_ED",
                                          start_year=1750, end_year=1951, epoch_length=100,
                                          new_periods_column_name="periods100a")



matrix_obj = DTM(data_matrix_filepath = tdm_infile_path, metadata_df= metadata_df_periods)
# matrix_obj = matrix_obj.reduce_to(semanticword_list)
matrix_obj = matrix_obj.eliminate(stopword_list)
matrix_obj = matrix_obj.add_metadata(["Gattungslabel_ED", "periods100a"])


cat_labels = ["N", "E", "0E", "M", "R"]
matrix_obj = matrix_obj.reduce_to_categories("Gattungslabel_ED", cat_labels)
# matrix_obj.eliminate(["Gattungslabel_ED"])

matrix_obj.save_csv(dtm_outfile_path)



