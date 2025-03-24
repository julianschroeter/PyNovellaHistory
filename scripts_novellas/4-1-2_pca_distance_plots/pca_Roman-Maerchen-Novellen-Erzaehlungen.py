from clustering.my_pca import PC_df
from preprocessing.presetting import vocab_lists_dicts_directory, local_temp_directory ,global_corpus_representation_directory, load_stoplist, global_corpus_raw_dtm_directory
from preprocessing.metadata_transformation import full_genre_labels
from preprocessing.corpus import DTM
from preprocessing.sampling import sample_n_from_cat
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

system =  "my_xps" # "wcph113" # "my_mac" # "wcph104"
lang = "en" # "de"


colors_list = ["red", "green", "blue", "orange", "cyan"]

#colors_list = random.sample(colors_list, len(colors_list))
filename = "RFECV_red-to-381_LRM-R-N-E-0E-XEno-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" # "raw_dtm_l1_lemmatized_use_idf_False2500mfw.csv" #"no-stopwords_no-names_RFECV_red-to-515_LRM-R-N-E-0E-XEscaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" # "raw_dtm_non-lemmatized_abs_10000mfw.csv"# "raw_dtm_l2__use_idf_True1000mfw.csv" # "red-to-2500mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" #   "raw_dtm_lemmatized_l1_10000mfw.csv" # "raw_dtm_lemmatized_abs_10000mfw.csv" #    #
dtm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), filename)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

dtm_object = DTM(data_matrix_filepath =dtm_infile_path, metadata_csv_filepath=metadata_filepath)
dtm_object = dtm_object.add_metadata(["Gattungslabel_ED_normalisiert", "Nachname", "Medientyp_ED"])

#dtm_object = dtm_object.reduce_to_categories("Medientyp_ED", ["Rundschau", "Familienblatt"])


labels_list = ["R", "M"]


dtm_object = dtm_object.eliminate(["Medientyp_ED"])
#dtm_object = dtm_object.eliminate(["Gattungslabel_ED_normalisiert"])
dtm_object_red = dtm_object.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=labels_list)

input_df = dtm_object_red.data_matrix_df
input_df = sample_n_from_cat(input_df)


replace_dict = {"in_Deutscher_Novellenschatz": {True: "Novellenschatz", "0": "sonstige MLP", False: "sonstige MLP"}}


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "novellas (merged)", "E": "novellas (merged)",
                                                  "0E": "novellas (merged)",
                                    "R": "Roman", "M": "Märchen", "XE": "novellas (merged)"}}
replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "Erzählprosa",
                                    "R": "Roman", "M": "Märchen", "XE": "Erzählprosa"}}



input_df = full_genre_labels(input_df, replace_dict=replace_dict)




pc_df = PC_df(input_df=input_df)

pc_df.generate_pc_df(n_components=0.95)

# remove outliers
#pc_df.pc_target_df =  pc_df.pc_target_df.drop(pc_df.pc_target_df["PC_1"].idxmax())
#pc_df.pc_target_df = pc_df.pc_target_df.drop(pc_df.pc_target_df["PC_2"].idxmax())

print(pc_df.pc_target_df.sort_values(by=["PC_2"], axis=0, ascending=False))

print(pc_df.component_loading_df.iloc[:,0].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[:,1].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[:,0].sort_values(ascending=True)[:20])
print(pc_df.component_loading_df.loc[:,1].sort_values(ascending=True)[:20])
print(pc_df.pca.explained_variance_)

ist_targetlabels = ", ".join(map(str, set(pc_df.pc_target_df["target"].values))).split(", ")

zipped_dict = {"Roman":"blue","Märchen":"orange" }
print(zipped_dict)
list_target = list(pc_df.pc_target_df["target"].values)
print(list_target)

# wenn ein Label, z.B. "other" farblos (weiß) werden soll:
# zipped_dict["other"] ="white"

title = "Romane vs. Märchen"

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
colors_str = ", ".join(map(str, pc_df.pc_target_df["target"].values))
colors_str = colors_str.translate(zipped_dict)
list_targetcolors = [zipped_dict[label] for label in list_target]

axes[0].scatter(pc_df.pc_target_df.iloc[:, 0], pc_df.pc_target_df.iloc[:, 1], c=list_targetcolors, cmap='rainbow',
            alpha=0.8)
if lang == "en":
    axes[0].set_xlabel(str('First Component, var expl.: ' + str(round(pc_df.pca.explained_variance_ratio_[0], 2))))
    axes[0].set_ylabel(str('Second Component, var. expl: ' + str(round(pc_df.pca.explained_variance_ratio_[1], 2))))

elif lang == "de":
    axes[0].set_xlabel(str('Erste Komponente, erklärte Varianz: ' + str(round(pc_df.pca.explained_variance_ratio_[0], 2)).replace(".", ",")))
    axes[0].set_ylabel(str('Zweite Komponente, erklärte Varianz: ' + str(round(pc_df.pca.explained_variance_ratio_[1], 2)).replace(".", ",")))

# plt.xscale(value="log")
# plt.yscale(value="log")
# plt.xlim(-50000,500000)
axes[0].set_title(title)
mpatches_list = []
for key, value in zipped_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)
axes[0].legend(handles=mpatches_list)

# now for N vs E:
labels_list = ["N", "E"]
dtm_object = dtm_object.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=labels_list)

input_df = dtm_object.data_matrix_df
input_df = sample_n_from_cat(input_df)


replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "Erzählprosa",
                                    "R": "Roman", "M": "Märchen", "XE": "Erzählprosa"}}



input_df = full_genre_labels(input_df, replace_dict=replace_dict)




pc_df = PC_df(input_df=input_df)
pc_df.generate_pc_df(n_components=0.95)

# remove outliers
#pc_df.pc_target_df =  pc_df.pc_target_df.drop(pc_df.pc_target_df["PC_1"].idxmax())
#pc_df.pc_target_df = pc_df.pc_target_df.drop(pc_df.pc_target_df["PC_2"].idxmax())

print(pc_df.pc_target_df.sort_values(by=["PC_2"], axis=0, ascending=False))

print(pc_df.component_loading_df.iloc[:,0].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[:,1].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[:,0].sort_values(ascending=True)[:20])
print(pc_df.component_loading_df.loc[:,1].sort_values(ascending=True)[:20])
print(pc_df.pca.explained_variance_)

ist_targetlabels = ", ".join(map(str, set(pc_df.pc_target_df["target"].values))).split(", ")

zipped_dict = {"Novelle":"red","Erzählung":"green" }
print(zipped_dict)
list_target = list(pc_df.pc_target_df["target"].values)
print(list_target)

# wenn ein Label, z.B. "other" farblos (weiß) werden soll:
# zipped_dict["other"] ="white"

title = "Novellen vs. Erzählungen"


colors_str = ", ".join(map(str, pc_df.pc_target_df["target"].values))
colors_str = colors_str.translate(zipped_dict)
list_targetcolors = [zipped_dict[label] for label in list_target]

axes[1].scatter(pc_df.pc_target_df.iloc[:, 0], pc_df.pc_target_df.iloc[:, 1], c=list_targetcolors, cmap='rainbow',
            alpha=0.8)
if lang == "en":
    axes[1].set_xlabel(str('First Component, var expl.: ' + str(round(pc_df.pca.explained_variance_ratio_[0], 2))))
    axes[1].set_ylabel(str('Second Component, var. expl: ' + str(round(pc_df.pca.explained_variance_ratio_[1], 2))))

elif lang == "de":
    axes[1].set_xlabel(str('Erste Komponente, erklärte Varianz: ' + str(round(pc_df.pca.explained_variance_ratio_[0], 2)).replace(".", ",")))
    axes[1].set_ylabel(str('Zweite Komponente, erklärte Varianz: ' + str(round(pc_df.pca.explained_variance_ratio_[1], 2)).replace(".",",")))

# plt.xscale(value="log")
# plt.yscale(value="log")
# plt.xlim(-50000,500000)
axes[1].set_title(title)
mpatches_list = []
for key, value in zipped_dict.items():
    patch = mpatches.Patch(color=value, label=key)
    mpatches_list.append(patch)
axes[1].legend(handles=mpatches_list)



if lang == "en":
    fig.suptitle("Principal Component Analyses (PCA)")
elif lang == "de":
    fig.suptitle("Hauptkomponentenanalyse (PCA)")
fig.tight_layout()
filename=os.path.join(local_temp_directory(system), "figures", lang + "_genres_pca-R-M_and_N-E.svg")
fig.savefig(filename)
plt.show()
