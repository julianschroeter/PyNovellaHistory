system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

from preprocessing.presetting import global_corpus_raw_dtm_directory

scaler = StandardScaler()

for filename in os.listdir(global_corpus_raw_dtm_directory(system)):
    if filename == "raw_dtm_l1__use_idf_False2500mfw.csv":
        filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

        raw_dtm = pd.read_csv(filepath, index_col=0)
        scaled_dtm = pd.DataFrame(scaler.fit_transform(raw_dtm.to_numpy()), columns=raw_dtm.columns, index=raw_dtm.index)
        outfile_name = str("scaled_" + filename)
        outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), outfile_name)
        scaled_dtm.to_csv(path_or_buf=outfile_path)
