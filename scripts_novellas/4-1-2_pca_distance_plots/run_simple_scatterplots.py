from preprocessing.corpus import DTM, DocFeatureMatrix
from preprocessing.metadata_transformation import years_to_periods
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory, load_stoplist
from clustering.my_plots import scatter
import pandas as pd
import os
system = "wcph113" # "my_mac" "my_xps"

data_filepath = os.path.join(global_corpus_raw_dtm_directory(system), "DocThemesMatrix_genres.csv")
colors_list = load_stoplist(os.path.join(global_corpus_representation_directory(system), "my_colors.txt"))

df = pd.read_csv(data_filepath, index_col= 0)

print(df)

scatter(df, colors_list)