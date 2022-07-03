import pandas as pd
from preprocessing.presetting import local_temp_directory
import os

system = "wcph113"

df = pd.read_csv(os.path.join(local_temp_directory(system), "bootstr-class-results.csv"), index_col=0)

df = df.round({"mean_lr":2, "std_lr":2, "mean_svm":2, "std_svm":2})
df.to_csv(os.path.join(local_temp_directory(system), "bootstr-class-results.csv"))

print(df)