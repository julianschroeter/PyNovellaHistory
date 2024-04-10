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
filename = "for_women_authors_heatmap.csv"
filename = "1_cosineselectet_authors_for-heatmap.csv"

df = pd.read_csv(os.path.join(local_temp_directory(system), filename), index_col=0)
print(df.values)

#df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)

sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
            vmin=0.9, vmax=1.7)
plt.title("Heatmap für DQ zwischen Autoren vor 1850")
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures", "heatmap_author_distances.svg"))
plt.show()
