from preprocessing.text import Text
from preprocessing.corpus_alt import DocFeatureMatrix

import pandas as pd
import os


class FearShare(Text):
    def __init__(self, filepath,
                 text,
                 id,
                 chunks,
                 token_length,
                 pos_triples,
                 remove_hyphen,
                 normalize_orthogr,
                 normalization_table_path,
                 correct_ocr,
                 eliminate_pagecounts,
                 handle_special_characters,
                 inverse_translate_umlaute,
                 eliminate_pos_items=False,
                 keep_pos_items=False,
                 list_keep_pos_tags=None,
                 list_eliminate_pos_tags=None,
                 lemmatize=False,
                 sz_to_ss=False,
                 translate_umlaute=False,
                 reduce_to_words_from_list=False,
                 reduction_word_list=None,
                 max_length=1000000,
                 remove_stopwords=False,
                 stopword_list=None,
                 language_model=None):
        Text.__init__(self, filepath,
                      text,
                      id,
                      chunks,
                      pos_triples,
                      token_length,
                      remove_hyphen,
                      normalize_orthogr,
                      normalization_table_path,
                      correct_ocr,
                      eliminate_pagecounts,
                      handle_special_characters,
                      inverse_translate_umlaute,
                      eliminate_pos_items,
                      keep_pos_items,
                      list_keep_pos_tags,
                      list_eliminate_pos_tags,
                      lemmatize,
                      sz_to_ss,
                      translate_umlaute,
                      reduce_to_words_from_list,
                      reduction_word_list,
                      max_length,
                      remove_stopwords,
                      stopword_list,
                      language_model)
    def calculate_share(self, fear_dict, case_sensitive=False, normalize='l1', standardize=True):
        text_as_list = self.text.split(" ")
        shares = 0

        if case_sensitive == True:
                for token in text_as_list:
                    if token in fear_dict:
                        shares += fear_dict[token]
        elif case_sensitive == False:
            for token in text_as_list:
                token = token.lower()
                fear_dict = {k.lower(): v for k, v in fear_dict.items()}
                if token in fear_dict:
                    shares += fear_dict[token]

            if normalize == "abs" and standardize == False:
                shares = shares
            elif normalize == "l1" and standardize == False:
                shares = shares / len(text_as_list)
            elif normalize == "l1" and standardize == True:
                shares = shares / (len(text_as_list) * len(fear_dict))

        return shares

class DocSentFearMatrix(DocFeatureMatrix):
    def __init__(self, sent_dict,
                 corpus_path,
                 remove_hyphen,
                 normalize_orthogr,
                 normalization_table_path,
                 correct_ocr,
                 eliminate_pagecounts,
                 handle_special_characters,
                 inverse_translate_umlaute,
                 keep_pos_items,
                 eliminate_pos_items,
                 list_of_pos_tags,
                 list_eliminate_pos_tags,
                 lemmatize,
                 sz_to_ss,
                 translate_umlaute,
                 remove_stopwords,
                 language_model,
                 data_matrix_df=None, data_matrix_filepath=None, metadata_csv_filepath=None,
                 metadata_df=None, mallet=False,
                 corpus_as_dict=None):
        DocFeatureMatrix.__init__(self, data_matrix_df, data_matrix_filepath, metadata_csv_filepath, metadata_df,
                                  mallet)
        self.sent_dict = sent_dict
        self.corpus_path = corpus_path
        self.handle_special_characters = handle_special_characters
        self.normalize_orthogr = normalize_orthogr
        self.normalization_table_path = normalization_table_path
        self.correct_ocr = correct_ocr
        self.handle_special_characters = handle_special_characters
        self.inverse_translate_umlaute = inverse_translate_umlaute
        self.eliminate_pagecounts = eliminate_pagecounts
        self.eliminate_pos_items = eliminate_pos_items
        self.keep_pos_items = keep_pos_items
        self.list_of_pos_tags = list_of_pos_tags
        self.list_eliminate_pos_tags = list_eliminate_pos_tags
        self.lemmatize = lemmatize
        self.sz_to_ss = sz_to_ss
        self.translate_umlaute = translate_umlaute
        self.remove_hyphen = remove_hyphen
        self.remove_stopwords = remove_stopwords
        self.language_model = language_model
        self.corpus_as_dict = corpus_as_dict

        print("language model is: ", self.language_model)

        if self.corpus_as_dict is None:
            dic = {}
            for filepath in os.listdir(self.corpus_path):
                if filepath == filepath: # here you can define some restrictions
                    theme_shares_obj = FearShare(filepath=os.path.join(self.corpus_path, filepath), token_length=0,
                                                    keep_pos_items=self.keep_pos_items,
                                                    text=None, id=None, chunks=None,
                                                    pos_triples=None, remove_hyphen=True,
                                                    correct_ocr=self.correct_ocr, eliminate_pagecounts=self.eliminate_pagecounts,
                                                    handle_special_characters=self.handle_special_characters,
                                                    normalize_orthogr=self.normalize_orthogr, normalization_table_path=self.normalization_table_path,
                                                    inverse_translate_umlaute=self.inverse_translate_umlaute,
                                                    eliminate_pos_items=self.eliminate_pos_items,
                                                    list_keep_pos_tags=self.list_of_pos_tags,
                                                    list_eliminate_pos_tags=self.list_eliminate_pos_tags, lemmatize=self.lemmatize,
                                                    sz_to_ss=False, translate_umlaute=False, max_length=5000000,
                                                    remove_stopwords="before_chunking", stopword_list=None,
                                                    language_model=self.language_model)
                    theme_shares_obj()
                    print("currently proceeds text with id: ", theme_shares_obj.id)
                    shares_list = theme_shares_obj.calculate_share(fear_dict=self.sent_dict)
                    dic[theme_shares_obj.id] = shares_list
            print(dic)
            df = pd.DataFrame.from_dict(dic, orient="index", columns=["Angstempfinden"])
            self.data_matrix_df = df