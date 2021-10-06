import os
import pandas as pd
from Preprocessing.MetadataTransformation import years_to_periods, generate_media_dependend_genres, standardize_meta_data_medium
from Preprocessing.Presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory
from Preprocessing.Corpus import DTM
from Preprocessing.Presetting import load_stoplist

system = "my_mac"
tdm_infile_path = os.path.join(global_corpus_raw_dtm_directory(system), "raw_dtm_lemmatized_tfidf_10000mfw.csv")
dtm_dep_genre_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), "dep_genre_tdm.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
raw_metadata_df = pd.read_csv(filepath_or_buffer=metadata_filepath, index_col=0)
stopword_list_filepath = os.path.join(global_corpus_representation_directory(system), "stopwords_all.txt")
stopword_list = load_stoplist(stopword_list_filepath)


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
