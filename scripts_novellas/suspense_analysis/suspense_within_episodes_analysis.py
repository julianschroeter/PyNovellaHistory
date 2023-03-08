system = "wcph113"

import pandas as pd
import os
import matplotlib.pyplot as plt

from preprocessing.presetting import local_temp_directory
df = pd.read_csv(os.path.join(local_temp_directory(system), "AllChunksDangerFearCharacters_novellas_episodes_scaled.csv"), index_col=0)
print(df)
# drop all rows with string type values in chunk id
#df = df[df["chunk_id"].map(lambda x: "-" not in x )]

print(df)

df = df.rename(columns={"max_value":"Gefahrenlevel", "chunk_id":"Textabschnitt"})
df["Textabschnitt"] = df["Textabschnitt"].apply((lambda x: x + 1))

columns = df.columns.values.tolist()
print(columns)
variables = ['Gewaltverbrechen', 'Kampf', 'Entführung', 'Krieg', 'Spuk', 'Gefahrenlevel', 'embedding_Angstempfinden', 'Angstempfinden', 'UnbekannteEindr', 'Feuer']

for variable in variables:
    # for all chunks with value > 0
    df = df[df[variable] != 0]
    print(df)
    df.boxplot(by="Textabschnitt", column=variable)
    means = df.groupby("Textabschnitt").median()
    print(means)
    plt.plot(means.index.values.tolist(), means[variable])
    plt.show()