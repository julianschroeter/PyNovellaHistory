from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from preprocessing.presetting import local_temp_directory

system = "my_xps" # "wcph113"

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



    def generate_pc_df(self, n_components=0.95, whiten=False):
        self.pca = PCA(n_components=n_components, whiten=whiten)
        pcs = self.pca.fit_transform(np.array(self.input_df.iloc[:, :-1]))
        print("estimated number of comoponents: ", self.pca.n_components_)
        pc_df = pd.DataFrame(data=pcs)
        pc_df = pc_df.iloc[:, 0:2]
        pc_target_df = pd.concat([pc_df, self.input_df.iloc[:, -1:].reset_index()], axis=1).set_index("index")
        pc_target_df.columns = ["PC_1", "PC_2", "target"]
        pc_target_df.dropna(subset=["target"], axis=0, inplace=True)
        self.pc_target_df = pc_target_df
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        self.component_loading_df = pd.DataFrame(loadings, index=self.input_df.iloc[:, :-1].columns)


    def scatter(self, colors_list, lang="en", filename=os.path.join(local_temp_directory(system),"figures","pca.png" ), title="PCA"):
        list_targetlabels = ", ".join(map(str, set(self.pc_target_df["target"].values))).split(", ")

        zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))
        print(zipped_dict)
        list_target = list(self.pc_target_df["target"].values)
        print(list_target)

        # wenn ein Label, z.B. "other" farblos (weiß) werden soll:
        # zipped_dict["other"] ="white"

        colors_str = ", ".join(map(str, self.pc_target_df["target"].values))
        colors_str = colors_str.translate(zipped_dict)
        list_targetcolors = [zipped_dict[label] for label in list_target]
        plt.figure(figsize=(6, 6))
        plt.scatter(self.pc_target_df.iloc[:,0], self.pc_target_df.iloc[:,1], c=list_targetcolors, cmap='rainbow', alpha=0.8)
        if lang == "en":
            plt.xlabel(str('First Component, var expl.: ' + str(round(self.pca.explained_variance_ratio_[0], 2))))
            plt.ylabel(str('Second Component, var. expl: ' + str(round(self.pca.explained_variance_ratio_[1], 2))))
            plt.title("Principal Component Analyses (PCA)")
        elif lang == "de":
            plt.xlabel(str('Erste Komponente, erklärte Varianz: ' + str(round(self.pca.explained_variance_ratio_[0], 2))))
            plt.ylabel(str('Zweite Komponente, erklärte Varianz: ' + str(round(self.pca.explained_variance_ratio_[1], 2))))
            plt.title("Hauptkomponentenanalyse (PCA)")
        #plt.xscale(value="log")
        #plt.yscale(value="log")
        #plt.xlim(-50000,500000)
        plt.title(title)
        mpatches_list = []
        for key, value in zipped_dict.items():
            patch = mpatches.Patch(color=value, label=key)
            mpatches_list.append(patch)
        plt.legend(handles=mpatches_list)
        plt.savefig(filename)
        plt.show()


