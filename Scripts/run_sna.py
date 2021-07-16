from Preprocessing.SNA import CharacterNetwork
from Preprocessing.Text import Text
from Preprocessing.Presetting import global_corpus_directory
from Preprocessing.Presetting import global_corpus_representation_directory
from Preprocessing.Presetting import set_DistReading_directory
from Preprocessing.Corpus import DocNetworkfeature_Matrix

import os
import matplotlib.pyplot as plt

system = "my_mac" # wcph104" "my_xps"

my_model_de = os.path.join("/Users/karolineschroter/Documents/CLS/Sprachmodelle", "my_model_de")

corpus_path = global_corpus_directory(system, test=True)
outfile_path_df = os.path.join(global_corpus_representation_directory(system), "Networkdata_document_Matrix_test.csv")
outfile_path_characters_list = os.path.join(global_corpus_representation_directory(system), "corpus_characters_list.txt")

network_matrix_object = DocNetworkfeature_Matrix(corpus_path=corpus_path, language_model=my_model_de, segmentation_type="fixed")
network_matrix_object.generate_df()
network_matrix_object.generate_ch
network_matrix_object.save_csv(outfile_path= outfile_path_df)
print(network_matrix_object.corpus_characters_list)
network_matrix_object.corpus_characters_list_to_file(outfilepath=outfile_path_characters_list)