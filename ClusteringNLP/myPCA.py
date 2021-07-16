from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
from Preprocessing.Presetting import global_corpus_representation_directory, load_stoplist
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class PC_df():
    """
    input_df is a Data Frame with features in columns and the categorical target variable in the last column
    """
    def __init__(self, input_df=None, target=None, pc_df=None, pc_target_df = None, pca=None, component_loading_df=None):
        self.input_df = input_df
        self.target = target
        self.pc_df = pc_df
        self.pc_target_df = pc_target_df
        self.pca = pca
        self.component_loading_df = component_loading_df

    def load_df(self, input_df_filepath):
        self.input_df = pd.read_csv(input_df_filepath, index_col=0)



    def generate_pc_df(self, n_components=2, whiten=False):
        self.pca = PCA(n_components=n_components, whiten=whiten)
        pcs = self.pca.fit_transform(np.array(self.input_df.iloc[:, :-1]))
        pc_df = pd.DataFrame(data=pcs)
        pc_target_df = pd.concat([pc_df, self.input_df.iloc[:, -1:].reset_index()], axis=1).set_index("index")
        pc_target_df.columns = ["PC_1", "PC_2", "target"]
        pc_target_df.dropna(subset=["target"], axis=0, inplace=True)
        self.pc_target_df = pc_target_df
        self.component_loading_df = pd.DataFrame(self.pca.components_, columns=self.input_df.iloc[:, :-1].columns,
                                                 index=["PC_1", "PC_2"])


    def scatter(self, colors_list):
        list_targetlabels = ", ".join(map(str, set(self.pc_target_df["target"].values))).split(", ")

        zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))
        list_target = list(self.pc_target_df["target"].values)

        # wenn ein Label, z.B. "other" farblos (weiß) werden soll:
        # zipped_dict["other"] ="white"

        colors_str = ", ".join(map(str, self.pc_target_df["target"].values))
        colors_str = colors_str.translate(zipped_dict)
        list_targetcolors = [zipped_dict[label] for label in list_target]
        plt.figure(figsize=(6, 6))
        plt.title("PCA")
        plt.scatter(self.pc_target_df.iloc[:,0], self.pc_target_df.iloc[:,1], c=list_targetcolors, cmap='rainbow', alpha=0.8)
        plt.xlabel(str('Erste Komponente, Erklärte Varianz: ' + str(round(self.pca.explained_variance_ratio_[0], 2))))
        plt.ylabel(str('Zweite Komponente, Erklärte Varianz: ' + str(round(self.pca.explained_variance_ratio_[1], 2))))
        #plt.xscale(value="log")
        #plt.yscale(value="log")
        #plt.xlim(-50000,500000)
        mpatches_list = []
        for key, value in zipped_dict.items():
            patch = mpatches.Patch(color=value, label=key)
            mpatches_list.append(patch)
        plt.legend(handles=mpatches_list)
        plt.show()

