from preprocessing.corpus import DTM, DocFeatureMatrix
from preprocessing.metadata_transformation import years_to_periods
from preprocessing.presetting import global_corpus_representation_directory, global_corpus_raw_dtm_directory
from preprocessing.presetting import load_stoplist
import pandas as pd
import os
system = "wcph113" # "my_mac" "my_xps"


infile_path = os.path.join(global_corpus_representation_directory(system), "DocThemesMatrix.csv")
metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
dtm_outfile_path = os.path.join(global_corpus_raw_dtm_directory(system), "DocThemesMatrix_genres.csv")

#stopword_list_filepath = os.path.join(global_corpus_representation_directory(system), "stopwords_all.txt")
#stopword_list = load_stoplist(stopword_list_filepath)
#wordlist_german_filepath = os.path.join(global_corpus_representation_directory(system), "wordlist_german.txt")
#wordlist_german = load_stoplist(wordlist_german_filepath)
#print("Länge der Worliste Deutsch: ", len(wordlist_german))

dtm_object = DocFeatureMatrix(data_matrix_filepath =infile_path, metadata_csv_filepath=metadata_filepath)

#dtm_object, eliminated_terms_list = dtm_object.reduce_to(wordlist_german, return_eliminated_terms_list=True)
#print(eliminated_terms_list)
#dtm_object = dtm_object.eliminate(stopword_list)


dtm_object = dtm_object.add_metadata("Gattungslabel_ED_normalisiert")
cat_labels = ["N", "E", "0E", "R", "M", "XE"]
dtm_object = dtm_object.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)


dtm_object.save_csv(dtm_outfile_path)
new_df = dtm_object.data_matrix_df
print(new_df)


# initialize Corpus() object new and add textanalytic metadata file as new metadata file:
#corpus_object = TDM(corpus_df=corpus_object.corpus_df, metadata_csv_filepath=textanalytic_metadata_filepath)
#corpus_object.generate_corpus()
#corpus_object.add_metadata("Ende")
#corpus_object.corpus_df["Ende"].fillna("other", inplace=True)


#corpus_object.processing_df.replace({"Gattungslabel_ED": {"N": "Prosaerzählung",
 #                                                                                  "E": "Prosaerzählung",
  #                                                                                 "0E": "Prosaerzählung",
   #                                                                                "R": "Roman",
    #                                                                               "M": "Märchen"
     #                                                                              }}, inplace=True)


#corpus_object.corpus_df = corpus_object.corpus_df.drop(["Gattungslabel_ED"], axis=1)



