import pandas as pd
import os
from Preprocessing.Presetting import local_temp_directory, global_corpus_raw_dtm_directory

system = "wcph113"

infile_name = os.path.join(global_corpus_raw_dtm_directory(system), "raw_dtm_5000mfw_lemmatized_tfidf_5000mfw.csv")

df = pd.read_csv(infile_name)

print(df)