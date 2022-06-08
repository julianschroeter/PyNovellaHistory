
from preprocessing.presetting import local_temp_directory

import os
import pandas as pd

system = "wcph113"


for filename in os.listdir(local_temp_directory((system))):
    print(filename)
    if "1000_euclid" in filename and "RFE.csv"  in filename:
        filepath = os.path.join(local_temp_directory(system), filename)
        df_eucl = pd.read_csv(filepath, index_col=0)
    elif "1000_cosine" in filename and "RFE.csv"  in filename:
        filepath = os.path.join(local_temp_directory(system), filename)
        df_cosine = pd.read_csv(filepath, index_col=0)
    elif "1000_city" in filename and "RFE.csv" in filename:
        filepath = os.path.join(local_temp_directory(system), filename)
        df_manh = pd.read_csv(filepath, index_col=0)
new_df = pd.concat([df_cosine, df_eucl, df_manh], axis=1)

new_df.to_csv(os.path.join(local_temp_directory(system), "concat_dist_n1000_resultsRFE.csv"))

print(new_df)