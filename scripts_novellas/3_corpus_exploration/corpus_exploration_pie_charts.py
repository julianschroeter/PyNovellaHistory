import pandas as pd
import matplotlib.pyplot as plt
import os
from preprocessing.presetting import local_temp_directory

sytem = "my_xps"
df = pd.read_csv(os.path.join(local_temp_directory(sytem), "Korpusgroessee_Gattungen.csv")) # here, the filepath has to be set appropriately!

fig, ax = plt.subplots(figsize=(12,8))
ax.pie(df.iloc[:,1], labels=df.iloc[:,0],autopct='%1.1f%%',
       colors=["red", "green", "cyan", "blue","orange", "grey", "magenta"])
plt.title("Korpus nach Gattungen")
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(sytem), "figures", "Korpusgroesse_Gattungen.svg"))
plt.show()

df = pd.read_csv(os.path.join(local_temp_directory(sytem), "Korpusgroesse_Medienformate.csv")) # and here as well


fig, ax = plt.subplots()
ax.pie(df.iloc[:,1], labels=df.iloc[:,0],autopct='%1.1f%%',
       colors=["purple", "lightgreen", "lightblue", "grey","yellow", "darkgreen", "magenta"])
plt.title("Korpus nach Medienformaten")
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(sytem), "figures", "Korpusgroesse_Medienformate.svg"))
plt.show()


