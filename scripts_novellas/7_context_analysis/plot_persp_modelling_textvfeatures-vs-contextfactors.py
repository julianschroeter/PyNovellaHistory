import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/julian/Documents/Uni/01_Habil/03_Abbildungen/persp-hist-modellierung-N-vs-E_textfeat-vs-contextfactors_kap7.csv",
                 index_col=0)

df.plot(kind="line")
plt.xlabel("Zeit")
plt.ylabel("Vorhersagegenauigkeit")
plt.title("Persp. Modellierung: Textmerkmale versus Kontextfaktoren")
plt.xticks([0,1,2,3], ["1805", "1835", "1865",
                             "1895"])
plt.savefig("/home/julian/Documents/Uni/01_Habil/03_Abbildungen/persp-hist-modellierung-N-vs-E_textfeat-vs-contextfactors_kap7.svg")
plt.show()

