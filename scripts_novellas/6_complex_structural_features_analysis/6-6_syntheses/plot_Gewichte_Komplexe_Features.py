system = ("my_xps")
from preprocessing.presetting import local_temp_directory
import os
import pandas as pd
from matplotlib import pyplot as plt

filepath = os.path.join(local_temp_directory(system),"Gewichte_Komplexe_Features.csv")

df = pd.read_csv(filepath, index_col=0)
print(df)
df = df

axes = df.plot.bar(rot=45, stacked=False)
#axes[1].legend(loc=2)
plt.title("Gewichte der komplexen Merkmale")
plt.xlabel("Komplexe Merkmale")
plt.ylabel("Gewichte der komplexen Merkmale: \n Erzählung <------> Novellle")
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures", "Komplexe_Featuregewichte.svg"))
plt.show()


df.unstack().plot(kind='bar', color=["lightgreen", "lightcoral"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures", "Komplexe_Featuregewichte_alternativ.svg"))
plt.show()