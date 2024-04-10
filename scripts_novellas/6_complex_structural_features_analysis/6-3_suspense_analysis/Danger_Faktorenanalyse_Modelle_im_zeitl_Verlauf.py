system = "my_xps"  # "wcph113" #
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

from preprocessing.presetting import global_corpus_representation_directory, language_model_path, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import years_to_periods, full_genre_labels
import matplotlib.pyplot as plt
import os
from scipy import stats
import pandas as pd
import numpy as np
my_model_de = language_model_path(system)

infile_df_path = os.path.join(local_temp_directory(system), "Varianzaufklärung_G-M_fuer_Gefahrenlevel_1850-1900.csv")

df = pd.read_csv(infile_df_path, index_col=0)
print(df)

df.plot(kind="line")
plt.title("Varianzaufklärung Delta für Gefahrenlevel")
plt.ylabel("Delta")
plt.xlabel("Erscheinungszeitraum")
plt.savefig(os.path.join(local_temp_directory(system), "figures", "Abb_Modelleistung_Aufklärung_Danger.svg"))
plt.show()