system = "my_xps"
from preprocessing.presetting import local_temp_directory

import os
import matplotlib.pyplot as plt
import pandas as pd

filepath = os.path.join(local_temp_directory(system), "tables", "data_tables", "genre-pairs-contextual-modeling-single-factors.csv")

df = pd.read_csv(filepath, index_col=0)

print(df)

df = df.drop(columns=["Author and Time"])

df.T.plot(kind='bar', stacked=False, title= "Context model: \nSingle factors for genre pairs",
        color={"Novelle vs. Roman": "orange", "Novelle vs. Erzählung": "magenta", "Roman vs. Erzählung": "grey"})

#plt.xticks(ticks=[0.0,1.0,2.0], labels=df.columns.tolist(), rotation=45)
plt.xticks(ticks=[0.0,1.0], labels=df.columns.tolist(), rotation=45)
plt.ylim(0.5,1)
plt.ylabel("Accuracy")
plt.tight_layout()

plt.savefig(os.path.join(local_temp_directory(system), "figures", "context-perspectival_single-factors-genrepairs.svg"))
plt.show()