from clustering.my_pca import PC_df
from preprocessing.presetting import vocab_lists_dicts_directory, local_temp_directory ,global_corpus_representation_directory, load_stoplist, global_corpus_raw_dtm_directory
from preprocessing.metadata_transformation import full_genre_labels
from preprocessing.corpus import DTM
from preprocessing.sampling import sample_n_from_cat
import os
import random

system = "wcph113" # "my_mac" # "wcph104" #  "my_xps"


colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "my_colors.txt"))
colors_list = random.sample(colors_list, len(colors_list))

colors_list = ["red", "green", "blue", "orange", "cyan"]
colors_list = ["orange", "blue", "cyan","red", "green"]
#colors_list = random.sample(colors_list, len(colors_list))
filename = filename = "raw_dtm_l1_lemmatized_use_idf_False2500mfw.csv" # "red-to-2500mfw_scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv" #
dtm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), filename)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

dtm_object = DTM(data_matrix_filepath =dtm_infile_path, metadata_csv_filepath=metadata_filepath)
dtm_object = dtm_object.add_metadata(["Gattungslabel_ED_normalisiert", "Nachname", "Medientyp_ED"])

#dtm_object = dtm_object.reduce_to_categories("Medientyp_ED", ["Rundschau", "Familienblatt"])
dtm_object = dtm_object.eliminate(["Medientyp_ED"])


labels_list = ["R", "M", "E", "N", "0E", "XE"]
labels_list = ["E", "N"]
labels_list = ["N", "E"]
labels_list = ["R", "M"]
dtm_object = dtm_object.reduce_to_categories(metadata_category="Gattungslabel_ED_normalisiert", label_list=labels_list)
#dtm_object = dtm_object.eliminate(["Gattungslabel_ED_normalisiert"])
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
pc_df.scatter(colors_list, lang="en", filename=os.path.join(local_temp_directory(system), "figures","genre_pca.png"))


