from Preprocessing.Corpus import DTM, DocFeatureMatrix
from Preprocessing.MetadataTransformation import years_to_periods, generate_media_dependend_genres, standardize_meta_data_medium
from Preprocessing.Presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory
from Preprocessing.Presetting import load_stoplist
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

system = "my_mac" # "wcph104" "my_mac" "my_xps"

tdm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), "raw_dtm_lemmatized_tfidf_10000mfw.csv")
char_netw_infile_path = os.path.join(global_corpus_representation_directory(system), "Networkdata_document_Matrix.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "textanalytic_metadata.csv")
colors_list = load_stoplist(os.path.join(global_corpus_representation_directory(system), "my_colors.txt"))

semanticword_list_filepath = os.path.join(global_corpus_representation_directory(system), "semantic_word_list_corpus_vocab.txt")
stopword_list_filepath = os.path.join(global_corpus_representation_directory(system), "names.txt")
verb_list_filepath = os.path.join(global_corpus_representation_directory(system), "verbs_list_corpus_vocab.txt")
semanticword_list = load_stoplist(semanticword_list_filepath)
stopword_list = load_stoplist(stopword_list_filepath)

new_metadata_df_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), "metatdata_table_periods30a.csv")
dtm_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), "character_network_table_genrelabel.csv")

dtm_dep_genre_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), "dep_genre_tdm.csv")

raw_metadata_df = pd.read_csv(filepath_or_buffer=metadata_filepath, index_col=0)
standardized_metadata_df = standardize_meta_data_medium(raw_metadata_df, "Medium_ED")

media_dependent_genres_metadata_df = generate_media_dependend_genres(standardized_metadata_df)

matrix_object_media_dependent = DTM(data_matrix_filepath=tdm_infile_path, metadata_df=media_dependent_genres_metadata_df)
new_object = matrix_object_media_dependent
#new_object = new_object.reduce_to(semanticword_list)
new_object = new_object.eliminate(stopword_list)
new_object = new_object.add_metadata(["Gattungslabel_ED","medium_type", "dependent_genre"])

cat_labels = ["N", "E"]
new_object = new_object.reduce_to_categories("Gattungslabel_ED", cat_labels)
cat_labels = ["tb", "famblatt"]
new_object = new_object.reduce_to_categories("medium_type", cat_labels)
new_object = new_object.eliminate(["medium_type", "dependent_genre"])

print(new_object.data_matrix_df)
new_object.data_matrix_df.to_csv(dtm_dep_genre_outfile_path)






metadata_df_periods30a = years_to_periods(input_df=pd.read_csv(filepath_or_buffer=metadata_filepath, index_col=0), category_name="Jahr_ED",
                                          start_year=1760, end_year=1950, epoch_length=30,
                                          new_periods_column_name="periods30a")


metadata_df_medium = generate_media_dependend_genres(())

metadata_df_periods30a.to_csv(path_or_buf=new_metadata_df_outfile_path)
matrix_indep_var = DTM(data_matrix_filepath =tdm_infile_path, metadata_csv_filepath= char_netw_infile_path)
matrix_indep_var.load_data_matrix_file()
matrix_indep_var.reduce_to(semanticword_list)
matrix_indep_var.eliminate(stopword_list)
matrix_indep_var.load_metadata_file()
matrix_indep_var.add_metadata(["Anteil"])

print(matrix_indep_var.processing_df)


matrix_with_genre = DTM(data_matrix_df=matrix_indep_var.processing_df, metadata_df=metadata_df_periods30a)
#matrix_with_genre.load_metadata_file()


matrix_with_genre.add_metadata(["periods30a", "Gattungslabel_ED"])


cat_labels = ["N", "E", "0E"]
matrix_with_genre.reduce_to_categories("Gattungslabel_ED", cat_labels)
matrix_with_genre.eliminate(["Gattungslabel_ED"])




matrix_with_genre.save_csv(dtm_outfile_path)

df = matrix_with_genre.processing_df

#df = df.drop(columns=["Figuren", "Figurenanzahl", "Netzwerkdichte"])

print(df)


def scatter(df, colors_list):
    list_targetlabels = ", ".join(map(str, set(df["Gattungslabel_ED"].values))).split(", ")
    print(list_targetlabels)
    zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))
    list_target = list(df["Gattungslabel_ED"].values)
    print(zipped_dict)
    colors_str = ", ".join(map(str, df["Gattungslabel_ED"].values))
    colors_str = colors_str.translate(zipped_dict)
    list_targetcolors = [zipped_dict[label] for label in list_target]
    plt.figure(figsize=(10, 10))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=list_targetcolors, cmap='rainbow')
    plt.xlabel('Erste Komponente')
    plt.ylabel('Zweite Komponente')
    plt.xlim(-0.1,1.1)
    plt.ylim(0, 80000)
    mpatches_list = []
    for key, value in zipped_dict.items():
        patch = mpatches.Patch(color=value, label=key)
        mpatches_list.append(patch)
    plt.legend(handles=mpatches_list)
    plt.show()

#scatter(df, colors_list)