system = "my_xps"

import pandas as pd
import os
from preprocessing.presetting import local_temp_directory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
data = pd.read_csv(os.path.join(local_temp_directory(system), "nearest_farest_predecessors_followers.csv"))
print(data)
colors = ["red" if e == "N" else "cyan" for e in data["Genre"].values.tolist()]

data["Name_Titel"] = data.apply(lambda x: x["Name"] + " '" + x["Titel"] + "'", axis=1)
texts = data["Name_Titel"].values.tolist()
x_values = data["Jahr"].values.tolist()
y_values = data["Rang pred"].values.tolist()
y_values = [ e for e in y_values]




farest = data["Rang farest followers"].values.tolist()
print(farest)
farest =  [(10 - e) if e != 0 else 0 for e in farest]

sizes = data["Rang farest cont"].values.tolist()
sizes = [(10 - e) if e != 0 else 0 for e in sizes]
i = 0

genres_mpatches_list = []
for genre, color in {"Novelle":"red", "sonstige Prosa":"cyan"}.items():
    patch = mpatches.Patch(color=color, label=genre)
    genres_mpatches_list.append(patch)

plt.scatter(data["Jahr"], data["Rang pred"], color=colors)
for text in texts:
    plt.text(x_values[i], y_values[i], text)
    if farest[i] != 0:
        print(x_values[i], y_values[i], farest[i])
        plt.arrow(x_values[i], y_values[i], -5 * farest[i] ,0, width=0.3, color= colors[i] )

    if sizes[i] != 0:
        print(x_values[i], y_values[i], sizes[i])
        plt.arrow(x_values[i], y_values[i], 0 ,0.05 * sizes[i], width=0.3, color= colors[i] )

    i += 1
plt.title("Vorbildhafte Texte und Abgrenzungsverhalten")
plt.ylabel("Rang:Vorbildhaftigkeit")
plt.legend(handles=genres_mpatches_list, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures", "predecessors_followers.svg"))
plt.show()

