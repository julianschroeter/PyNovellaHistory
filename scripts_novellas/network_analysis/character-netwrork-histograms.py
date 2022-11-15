import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.presetting import load_stoplist
from preprocessing.corpus_alt import DTM
import os
import matplotlib.pyplot as plt

system = "my_mac"
char_netw_infile_path = os.path.join(global_corpus_representation_directory(system), "Networkdata_document_Matrix.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")


matrix_with_genre = DTM(data_matrix_filepath=char_netw_infile_path, metadata_csv_filepath=metadata_filepath)
matrix_with_genre = matrix_with_genre.add_metadata(["Gattungslabel_ED"])

char_netw_df = matrix_with_genre.data_matrix_df

nov_df = char_netw_df[char_netw_df["Gattungslabel_ED"] == "N"]
E_df = char_netw_df[char_netw_df["Gattungslabel_ED"] == "E"]
print(nov_df)

nov_hist = nov_df.hist(column=["Netzwerkdichte"], bins = 20)

histogram_filepath = os.path.join(local_temp_directory(system), "nov_histogram_network-degree-equal-1.png")
plt.savefig(histogram_filepath)

E_hist = E_df.hist(column=["Netzwerkdichte"], bins = 20)
histogram_filepath = os.path.join(local_temp_directory(system), "E_histogram_network-degree-equal-1.png")

plt.savefig(histogram_filepath)