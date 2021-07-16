from ClusteringNLP.myPCA import PC_df
from Preprocessing.Presetting import global_corpus_representation_directory, load_stoplist, global_corpus_raw_dtm_directory
import os

system = "my_mac" # "wcph104" #  "my_xps"
dtm_filename =   "dep_genre_tdm.csv"
dtm_filepath = os.path.join(global_corpus_raw_dtm_directory(system), dtm_filename)
colors_list = load_stoplist(os.path.join(global_corpus_representation_directory(system), "my_colors.txt"))


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
