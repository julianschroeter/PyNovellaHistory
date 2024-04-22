from clustering.my_pca import PC_df
from preprocessing.presetting import vocab_lists_dicts_directory, local_temp_directory ,global_corpus_representation_directory, load_stoplist, global_corpus_raw_dtm_directory
from preprocessing.metadata_transformation import full_genre_labels, generate_media_dependend_genres, years_to_periods
from preprocessing.corpus import DTM
import matplotlib.patches as mpatches

from preprocessing.sampling import sample_n_from_cat
import os
import random
#import umap
import numpy as np
import matplotlib.pyplot as plt

system = "my_xps" #"wcph113" # "my_mac" # "wcph104" #
lang = "de"


colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "my_colors.txt"))
colors_list = random.sample(colors_list, len(colors_list))
colors_list = ["red", "green", "cyan", "orange", "cyan", "yellow", "black", "magenta"]

#colors_list = random.sample(colors_list, len(colors_list))

infilename = "red-to-1000mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" #  "raw_dtm_l1_lemmatized_use_idf_False2500mfw.csv" #  "red-to-2500mfw_scaled_no-stopwords_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" "red-to-100mfw_scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" #

dtm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), infilename)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

dtm_object = DTM(data_matrix_filepath =dtm_infile_path, metadata_csv_filepath=metadata_filepath)
dtm_object = dtm_object.add_metadata(["Gattungslabel_ED_normalisiert", "Medientyp_ED", "Jahr_ED"])





labels_list = ["R", "M", "E", "N", "0E", "XE"]
labels_list = ["E", "N"]
labels_list = ["R", "M"]
labels_list = ["N", "E"]
labels_list = ["E", "N", "0E", "XE", "M", "R"]
dtm_object = dtm_object.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=labels_list)


dtm_object.data_matrix_df = years_to_periods(input_df=dtm_object.data_matrix_df, category_name="Jahr_ED",
                                          start_year=1750, end_year=1951, epoch_length=100,
                                          new_periods_column_name="periods100a")

dtm_object = dtm_object.eliminate(["Jahr_ED"])




# input_df = sample_n_from_cat(input_df)

replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "N", "E": "E", "0E": "0E",
                                     "XE": "XE"}}


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "Erzählprosa",
                                    "R": "Roman", "M": "Märchen", "XE": "Erzählprosa"}}

dtm_object.data_matrix_df = full_genre_labels(dtm_object.data_matrix_df, replace_dict=replace_dict)

# reduce to several genres
dtm_object = dtm_object.reduce_to_categories("Gattungslabel_ED_normalisiert", ["Erzählung", "Novelle", "Erzählprosa"])

df = dtm_object.data_matrix_df

# reduce to a sub-period
#df = df[df["periods100a"] == "1750-1850"]
df = df.drop(columns=["periods100a"])

# reduce to several media types
media_labels_list = ["Familienblatt", "Rundschau","Taschenbuch", "Anthologie"] #, , "Zeitschrift"
df = df[df.isin({"Medientyp_ED": media_labels_list}).any(axis=1)]

# work with genre df:
genre_df = df.drop(columns=["Medientyp_ED"])

pc_df = PC_df(input_df=genre_df)
pc_df.generate_pc_df(n_components=2)
target_df = pc_df.pc_target_df

# drop outliers
pc_df.pc_target_df =  pc_df.pc_target_df.drop(pc_df.pc_target_df["PC_1"].idxmax())
pc_df.pc_target_df = pc_df.pc_target_df.drop(pc_df.pc_target_df["PC_2"].idxmax())

print(pc_df.pc_target_df.sort_values(by=["PC_2"], axis=0, ascending=False))
print(pc_df.component_loading_df.iloc[:,0].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[:,1].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[:,0].sort_values(ascending=True)[:20])
print(pc_df.component_loading_df.loc[:,1].sort_values(ascending=True)[:20])
print(pc_df.pca.explained_variance_)

# generate genre coloring
list_targetlabels = ", ".join(map(str, set(pc_df.pc_target_df["target"].values))).split(", ")
list_targetlabels = ["Novelle", "Erzählung", "Erzählprosa"]
zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))
list_target = list(pc_df.pc_target_df["target"].values)
colors_str = ", ".join(map(str, pc_df.pc_target_df["target"].values))
colors_str = colors_str.translate(zipped_dict)
list_targetcolors = [zipped_dict[label] for label in list_target]

fig, axes = plt.subplots(1, 2, constrained_layout=False, figsize=(10,5))

axes[0].scatter(pc_df.pc_target_df.iloc[:,0], pc_df.pc_target_df.iloc[:,1], c=list_targetcolors, cmap='rainbow', alpha=0.8)

mpatches_list = []
for key, value in zipped_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)
axes[0].legend(handles=mpatches_list)


# generate media coloring (right plot)
media_df = df.drop(columns=["Gattungslabel_ED_normalisiert"])
pc_df = PC_df(input_df=media_df)
pc_df.generate_pc_df(n_components=2)
target_df = pc_df.pc_target_df

# drop outliers
pc_df.pc_target_df =  pc_df.pc_target_df.drop(pc_df.pc_target_df["PC_1"].idxmax())
pc_df.pc_target_df = pc_df.pc_target_df.drop(pc_df.pc_target_df["PC_2"].idxmax())

colors_list = ["lightgreen", "grey","purple", "yellow"]
list_targetlabels = ", ".join(map(str, set(pc_df.pc_target_df["target"].values))).split(", ")
list_targetlabels = media_labels_list
zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))
list_target = list(pc_df.pc_target_df["target"].values)
colors_str = ", ".join(map(str, pc_df.pc_target_df["target"].values))
colors_str = colors_str.translate(zipped_dict)
list_targetcolors = [zipped_dict[label] for label in list_target]

axes[1].scatter(pc_df.pc_target_df.iloc[:,0], pc_df.pc_target_df.iloc[:,1], c=list_targetcolors, cmap='rainbow', alpha=0.8)

mpatches_list = []
for key, value in zipped_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)
axes[1].legend(handles=mpatches_list)

if lang == "en":
    plt.xlabel(str('First Component, var expl.: ' + str(round(pc_df.pca.explained_variance_ratio_[0], 2))))
    plt.ylabel(str('Second Component, var. expl: ' + str(round(pc_df.pca.explained_variance_ratio_[1], 2))))
    plt.suptitle("Principal Component Analyses (PCA)")
elif lang == "de":
    #plt.xlabel(str('Erste Komponente, erklärte Varianz: ' + str(round(pc_df.pca.explained_variance_ratio_[0], 2))))
    #plt.ylabel(str('Zweite Komponente, erklärte Varianz: ' + str(round(pc_df.pca.explained_variance_ratio_[1], 2))))
    plt.suptitle("Hauptkomponentenanalyse (PCA)")
    fig.text(0.5, 0.04, str('Erste Komponente, erklärte Varianz: ' + str(round(pc_df.pca.explained_variance_ratio_[0], 2)).replace(".",",")), ha='center')
    fig.text(0.04, 0.5, str('Zweite Komponente, erklärte Varianz: ' + str(round(pc_df.pca.explained_variance_ratio_[1], 2)).replace(".",",")), va='center', rotation='vertical')


filename= os.path.join(local_temp_directory(system),"figures", "PCA_media-types-and-genres.svg")
plt.savefig(filename)
plt.show()