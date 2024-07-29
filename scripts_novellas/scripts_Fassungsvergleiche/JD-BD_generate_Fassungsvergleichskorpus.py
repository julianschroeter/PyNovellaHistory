system = "my_xps"

from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory, global_corpus_raw_dtm_directory, global_corpus_directory
from preprocessing.corpus import DocFeatureMatrix, generate_text_files
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

corpus_path = global_corpus_directory(system)
metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

metadata_df = pd.read_csv(metadata_filepath, index_col=0)
print(metadata_df)
df = metadata_df.copy()
df = df.drop(labels="00311-00")
BD_df = df[df["spätere_Fassung_von"].notna()]

# drop works by author "Poe"
BD_df = BD_df[BD_df["Nachname"] != "Poe"]
BD_ids = BD_df.index.values.tolist()
ED_ids = BD_df["spätere_Fassung_von"].values.tolist()
# FV: Fassungsvergleiche
FV_ids = ED_ids + BD_ids

generate_text_files(chunking=False, pos_representation=False,corpus_path=corpus_path, outfile_path= os.path.join(local_temp_directory(system), "fassungsvergleich_sample"), only_selected_files=True, list_of_file_ids=FV_ids)

