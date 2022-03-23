from clustering.my_pca import PC_df
from preprocessing.presetting import vocab_lists_dicts_directory, local_temp_directory ,global_corpus_representation_directory, load_stoplist, global_corpus_raw_dtm_directory
import os

system = "wcph113" # "my_mac" # "wcph104" #  "my_xps"

dtm_filepath = os.path.join(global_corpus_raw_dtm_directory(system), "dep_genre_tdm.csv")
colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "my_colors.txt"))
print(colors_list)

pc_df = PC_df()
pc_df.load_df(dtm_filepath)
pc_df.generate_pc_df(n_components=2)


print(pc_df.pc_target_df.sort_values(by=["PC_2"], axis=0, ascending=False))
print(pc_df.component_loading_df.loc["PC_1"].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.loc["PC_2"].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.loc["PC_1"].sort_values(ascending=True)[:20])
print(pc_df.component_loading_df.loc["PC_2"].sort_values(ascending=True)[:20])
print(pc_df.pca.explained_variance_)
pc_df.scatter(colors_list)

import matplotlib.pyplot as plt
plt.savefig(os.path.join(local_temp_directory(system), "PCA.png"))