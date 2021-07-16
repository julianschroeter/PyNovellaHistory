from Preprocessing.Corpus import DTM, DocFeatureMatrix
from Preprocessing.MetadataTransformation import years_to_periods
from Preprocessing.Presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory
from Preprocessing.Presetting import load_stoplist
import pandas as pd
import os
system = "my_mac" # "my_xps"
infile_name = "raw_dtm_lemmatized_tfidf_10000mfw.csv"

dtm_infile_path= os.path.join(global_corpus_raw_dtm_directory(system), infile_name)
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "textanalytic_metadata.csv")
semanticword_list_filepath = os.path.join(global_corpus_representation_directory(system), "semantic_word_list_corpus_vocab.txt")
stopword_list_filepath = os.path.join(global_corpus_representation_directory(system), "stopwords_all.txt")
verb_list_filepath = os.path.join(global_corpus_representation_directory(system), "verbs_list_corpus_vocab.txt")
semanticword_list = load_stoplist(semanticword_list_filepath)
stopword_list = load_stoplist(stopword_list_filepath)
verb_list = load_stoplist(verb_list_filepath)

dtm_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), "dtm_tfidf_genrelabel.csv")



corpus_object = DTM(data_matrix_filepath =dtm_infile_path, metadata_df= metadata_df_periods30a)
corpus_object.load_data_matrix_file()
#corpus_object.load_metadata_file()
corpus_object.reduce_to(semanticword_list)
corpus_object.eliminate(stopword_list)
#corpus_object.eliminate(verb_list)
corpus_object.add_metadata("periods30a")

# initialize Corpus() object new and add textanalytic metadata file as new metadata file:

#corpus_object = TDM(corpus_df=corpus_object.corpus_df, metadata_csv_filepath=textanalytic_metadata_filepath)
#corpus_object.generate_corpus()
#corpus_object.add_metadata("Ende")
#corpus_object.corpus_df["Ende"].fillna("other", inplace=True)

#cat_labels = ["N", "E", "0E", "R", "M"]
#corpus_object.reduce_to_categories("Gattungslabel_ED", cat_labels)

corpus_object.processing_df.replace({"Gattungslabel_ED": {"N": "Prosaerzählung",
                                                                                   "E": "Prosaerzählung",
                                                                                   "0E": "Prosaerzählung",
                                                                                   "R": "Roman",
                                                                                   "M": "Märchen"
                                                                                   }}, inplace=True)


#corpus_object.corpus_df = corpus_object.corpus_df.drop(["Gattungslabel_ED"], axis=1)


print(corpus_object.processing_df)

corpus_object.save_csv(dtm_outfile_path)