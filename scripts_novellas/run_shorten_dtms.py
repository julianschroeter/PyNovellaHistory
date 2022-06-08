system = "wcph113"

from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

from preprocessing.presetting import global_corpus_raw_dtm_directory


for filename in os.listdir(global_corpus_raw_dtm_directory(system)):
    if "scaled_no-stopwords_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" in filename:
        filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)
        raw_dtm = pd.read_csv(filepath, index_col=0)
        n = 2500
        short_dtm = raw_dtm.iloc[: , : n]
        outfile_name = str("red-to-" + str(n)+ "mfw_" + filename)
        outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), outfile_name)
        short_dtm.to_csv(path_or_buf=outfile_path)

        n = 1000
        short_dtm = raw_dtm.iloc[:, : n]
        outfile_name = str("red-to-" + str(n) + "mfw_" + filename)
        outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), outfile_name)
        short_dtm.to_csv(path_or_buf=outfile_path)

        n = 500
        short_dtm = raw_dtm.iloc[:, : n]
        outfile_name = str("red-to-" + str(n) + "mfw_" + filename)
        outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), outfile_name)
        short_dtm.to_csv(path_or_buf=outfile_path)

        n = 100
        short_dtm = raw_dtm.iloc[:, : n]
        outfile_name = str("red-to-" + str(n) + "mfw_" + filename)
        outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), outfile_name)
        short_dtm.to_csv(path_or_buf=outfile_path)