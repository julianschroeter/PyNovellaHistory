system = "my_xps"

import os
import pandas as pd
from preprocessing.presetting import global_corpus_directory, global_corpus_representation_directory

conll_filepath = os.path.join(global_corpus_representation_directory(system), "conll_toponym_share_Matrix.csv")
conll_df = pd.read_csv(conll_filepath, index_col=0)
conll_df.fillna(0, inplace=True)
conll_df = conll_df.rename(columns={"rom_top":"rom_top_conll"})

spacy_filepath = os.path.join(global_corpus_representation_directory(system), "toponym_share_Matrix.csv")
df = pd.read_csv(spacy_filepath, index_col=0)
df.fillna(0, inplace=True)

new_df = pd.concat([conll_df, df], axis=1)
new_df.fillna(0, inplace=True)
new_df["rom_top_comb"] = new_df.apply(lambda x: (x.rom_top_conll + x.rom_top) / 2, axis=1)
new_df = new_df.loc[:, ["rom_top_comb"]]
new_df = new_df.rename(columns={"rom_top_comb":"rom_top"})
print(new_df)
new_df.to_csv(os.path.join(global_corpus_representation_directory(system), "rom_top_matrix_comb.csv"))