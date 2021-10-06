from Preprocessing.MetadataTransformation import extract_file_ids_period
import pandas as pd
import os
from Preprocessing.Presetting import global_corpus_representation_directory, global_corpus_directory, local_temp_directory
from Preprocessing.Corpus import generate_text_files
system = "my_mac" # "wcph104" "my_mac" "my_xps"
my_model = os.path.join("/Users/karolineschroter/Documents/CLS/Sprachmodelle", "my_model_de")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

input_directory = global_corpus_directory(system_name=system)
output_directory = os.path.join(local_temp_directory(system_name=system), "testlauf")

metadata_df = pd.read_csv(metadata_filepath, index_col=0)
print(metadata_df["Jahr_ED"])
ids_list = extract_file_ids_period(df=metadata_df,from_year=0, to_year=1830)
print(ids_list)
ids_list = ["00136-00", "00033-00", "00155-00"]
generate_text_files(list_of_file_ids=ids_list, chunking=False, outfile_path=output_directory, pos_representation=False, corpus_path=input_directory,
                    language_model=my_model)