from Preprocessing.Presetting import global_corpus_representation_directory, global_corpus_directory
from Preprocessing.Corpus import Corpus_TDM
import os

outfile_dtm_path = os.path.join(global_corpus_representation_directory("my_mac"), "raw_dtm")
corpus_path = global_corpus_directory("my_mac", test=False)
n_mfw = 10000
metadata_filepath = os.path.join(global_corpus_representation_directory("my_mac"), "Bibliographie.csv")

normalization_list = ["abs", "l1", "l2", "tfidf"]
lemmatization_list = "True"

for normalization in normalization_list:
    # for lemmatization == True
    raw_corpus_object = Corpus_TDM(corpus_path=corpus_path, dehyphen=True, lemmatize=True, correct_ocr=True, eliminiate_pagecounts=True, inverse_translate_umlaute=True, n_mfw=n_mfw,
                                   normalize=normalization)
    outfile_basename = "raw_dtm_lemmatized_"+str(normalization)+"_"+str(n_mfw)+"mfw.csv"
    raw_dtm_outfile_path = os.path.join(outfile_dtm_path, outfile_basename)
    raw_corpus_object.generate_corpus()
    raw_corpus_object.save_csv(raw_dtm_outfile_path)

    # for lemmatization == False
    raw_corpus_object = Corpus_TDM(corpus_path=corpus_path, dehyphen=True, lemmatize=False, correct_ocr=True, eliminiate_pagecounts=True, inverse_translate_umlaute=True, n_mfw=n_mfw,
                                   normalize=normalization)
    outfile_basename = "raw_dtm_non-lemmatized_" + str(normalization) + "_" + str(n_mfw) + "mfw.csv"
    raw_dtm_outfile_path = os.path.join(outfile_dtm_path, outfile_basename)
    raw_corpus_object.generate_corpus()
    raw_corpus_object.save_csv(raw_dtm_outfile_path)


