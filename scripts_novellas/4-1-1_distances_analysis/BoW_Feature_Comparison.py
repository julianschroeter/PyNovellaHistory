system = "my_xps"

import pandas as pd

system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')



from metrics.distances import DistResults, GroupDistances, results_2groups_dist
from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory, local_temp_directory
from preprocessing.presetting import local_temp_directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df = pd.read_csv("/home/julian/Documents/Uni/01_Habil/03_Abbildungen/tabelle_mfw-DistQuot-red.csv", index_col=0)

df = df.drop(columns=["Unnamed: 1"])
print(df)

df.plot(kind='bar')
plt.title("Evaluation der Featuresets im BoW-Modell")
plt.ylim(0.99, 1.05)
plt.hlines(y=1, xmin=0, xmax=12, color="black")
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures", "BoW_Feature_Comparison.svg"))
plt.show()
