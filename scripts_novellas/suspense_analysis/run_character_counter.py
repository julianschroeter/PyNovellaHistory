system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os
import pandas as pd
from preprocessing.presetting import local_temp_directory, global_corpus_representation_directory
from collections import Counter

corpus_path = os.path.join(local_temp_directory(system), "Namen_novellas_episodes_chunks")

d = {}
for filename in os.listdir(corpus_path):
    filepath = os.path.join(corpus_path, filename)
    names_text = open(filepath, "r", encoding="latin-1").read()
    names_list = names_text.split(" ")
    names_counter = Counter(names_list)
    print(names_counter)
    counts_list = [key for key, _ in names_counter.most_common()]
    d[filename] = counts_list[0]

df = pd.DataFrame.from_dict(d, orient="index")

print(df)


df.to_csv(os.path.join(local_temp_directory(system), "DocNamesCounterMatrix_novellas_episodes.csv"))
