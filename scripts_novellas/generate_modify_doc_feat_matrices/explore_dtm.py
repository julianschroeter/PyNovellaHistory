import os
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory
from preprocessing.corpus import DTM
system = "wcph113"

infile_name = os.path.join(global_corpus_raw_dtm_directory(system), "RFE_reduced_dtmLR.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

dtm_object = DTM(data_matrix_filepath=infile_name, metadata_csv_filepath=metadata_filepath)

print(dtm_object.data_matrix_df)

dtm_object = dtm_object.add_metadata(["Gattungslabel_ED_normalisiert"])

print(dtm_object.data_matrix_df)