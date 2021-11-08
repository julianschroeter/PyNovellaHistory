from preprocessing.corpus import DTM, DocFeatureMatrix
from preprocessing.metadata_transformation import years_to_periods
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory
from preprocessing.presetting import load_stoplist
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

system = "my_mac" # "wcph104" "my_mac" "my_xps"

tdm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), "raw_dtm_lemmatized_tfidf_10000mfw.csv")
char_netw_infile_path = os.path.join(global_corpus_representation_directory(system), "Networkdata_document_Matrix.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "textanalytic_metadata.csv")
colors_list = load_stoplist(os.path.join(global_corpus_representation_directory(system), "my_colors.txt"))

semanticword_list_filepath = os.path.join(global_corpus_representation_directory(system), "semantic_word_list_corpus_vocab.txt")
stopword_list_filepath = os.path.join(global_corpus_representation_directory(system), "stopwords_all.txt")
verb_list_filepath = os.path.join(global_corpus_representation_directory(system), "verbs_list_corpus_vocab.txt")
semanticword_list = load_stoplist(semanticword_list_filepath)
stopword_list = load_stoplist(stopword_list_filepath)

new_metadata_df_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), "metatdata_table_periods30a.csv")
dtm_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), "character_network_table_genrelabel.csv")

metadata_df_periods30a = years_to_periods(input_df=pd.read_csv(filepath_or_buffer=metadata_filepath, index_col=0), category_name="Jahr_ED",
                                          start_year=1760, end_year=1950, epoch_length=30,
                                          new_periods_column_name="periods30a")


metadata_df_periods50a = years_to_periods(input_df=metadata_df_periods30a, category_name="Jahr_ED",
                                          start_year=1750, end_year=1950, epoch_length=50,
                                          new_periods_column_name="periods100a")


metadata_df_periods30a["KanonStatus"].replace(np.nan, 0, inplace=True)
metadata_df_periods30a["KanonStatus"].replace(0, "nicht", inplace=True)
metadata_df_periods30a["KanonStatus"].replace(1, "schwach", inplace=True)
metadata_df_periods30a["KanonStatus"].replace(2, "mittel", inplace=True)
metadata_df_periods30a["KanonStatus"].replace(3, "stark", inplace=True)


metadata_df_periods30a.to_csv(path_or_buf=new_metadata_df_outfile_path)
matrix_indep_var = DTM(data_matrix_filepath =char_netw_infile_path, metadata_df=metadata_df_periods30a)
matrix_indep_var.load_data_matrix_file()

matrix_indep_var.eliminate(["Figuren", "Figurenanzahl", "Anteil"])
matrix_indep_var.add_metadata(["periods30a", "Gattungslabel_ED", "KanonStatus", "periods100a"])


cat_labels = ["N", "E", "0E"]
matrix_indep_var.reduce_to_categories("Gattungslabel_ED", cat_labels)
matrix_indep_var.eliminate(["KanonStatus", "periods100a", "periods30a"])




matrix_indep_var.save_csv(dtm_outfile_path)

df = matrix_indep_var.processing_df


print(df)


def scatter(df, colors_list):
    list_targetlabels = ", ".join(map(str, set(df.iloc[:,-1].values))).split(", ")
    print(list_targetlabels)
    zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))
    list_target = list(df.iloc[:,-1].values)
    print(zipped_dict)
    colors_str = ", ".join(map(str, df.iloc[:,-1].values))
    colors_str = colors_str.translate(zipped_dict)
    list_targetcolors = [zipped_dict[label] for label in list_target]
    plt.figure(figsize=(10, 10))
    plt.scatter(df.iloc[:, 1], df.iloc[:, 0], c=list_targetcolors, cmap='rainbow', alpha=0.5)
    plt.xlabel('Erste Komponente')
    plt.ylabel('Zweite Komponente')
    plt.ylim(-0.1,1.1)
    plt.xlim(0, 120000)
    mpatches_list = []
    for key, value in zipped_dict.items():
        patch = mpatches.Patch(color=value, label=key)
        mpatches_list.append(patch)
    plt.legend(handles=mpatches_list)
    plt.show()

scatter(df, colors_list)