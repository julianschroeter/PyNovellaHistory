import os
import pandas as pd

from preprocessing.text import Text
from preprocessing.corpus import DocFeatureMatrix

class SettingShare(Text):
    def __init__(self, filepath, text, id, chunks, token_length, pos_triples,
                 remove_hyphen, normalize_orthogr, normalization_table_path,
                 correct_ocr, eliminate_pagecounts, handle_special_characters,
                 inverse_translate_umlaute,
                 eliminate_pos_items, keep_pos_items, list_keep_pos_tags,
                 list_eliminate_pos_tags, lemmatize,
                 sz_to_ss, translate_umlaute, max_length,
                 remove_stopwords, stopword_list, language_model):

        Text.__init__(self, filepath, text, id, chunks,token_length,  pos_triples,
                         remove_hyphen, normalize_orthogr, normalization_table_path,
                 correct_ocr, eliminate_pagecounts, handle_special_characters,
                         inverse_translate_umlaute,
                 eliminate_pos_items, keep_pos_items, list_keep_pos_tags,
                         list_eliminate_pos_tags, lemmatize,
                 sz_to_ss, translate_umlaute, max_length,
                 remove_stopwords,  stopword_list, language_model)

    def calculate_share(self, list_of_wordlists, normalize='l1', standardize=True, case_sensitive=False):
        """
        proceeds the calculation operation and stores the result wordlist_shares attribute.
        """
        text_as_list = self.text.split(" ")
        shares_list = []
        for wordlist in list_of_wordlists:
            hits = 0
            if case_sensitive == True:
                for token in text_as_list:
                    if token in wordlist:
                        hits += 1
            elif case_sensitive == False:
                wordlist_lower = [token.lower() for token in wordlist]
                for token in text_as_list:
                    if token.lower() in wordlist_lower:
                        hits += 1

            if normalize ==  "abs" and standardize == False:
                shares = hits
            elif normalize == "l1" and standardize == False:
                shares = hits / len(text_as_list)
            elif normalize == "l1" and standardize == True:
                shares = hits / (len(text_as_list) * len(wordlist))

            shares_list.append(shares)

        return shares_list


class DocThemesMatrix(DocFeatureMatrix):
    def __init__(self, list_of_wordlists, corpus_path,
                 remove_hyphen, normalize_orthogr, normalization_table_path,
                 correct_ocr, eliminate_pagecounts, handle_special_characters,
                 inverse_translate_umlaute,
                 keep_pos_items,
                 eliminate_pos_items, list_of_pos_tags,
                 list_eliminate_pos_tags, lemmatize,
                 sz_to_ss, translate_umlaute,
                 remove_stopwords, language_model,
                 data_matrix_df=None, data_matrix_filepath=None, metadata_csv_filepath=None,
                 metadata_df=None, mallet=False,
                 corpus_as_dict=None):
        DocFeatureMatrix.__init__(self, data_matrix_df, data_matrix_filepath, metadata_csv_filepath, metadata_df,
                                  mallet)
        self.list_of_wordlists = list_of_wordlists
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

        if self.corpus_as_dict is None:
            dic = {}
            for filepath in os.listdir(self.corpus_path):
                theme_shares_obj = SettingShare(filepath=os.path.join(self.corpus_path, filepath), token_length=0,
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
                shares_list = theme_shares_obj.calculate_share(list_of_wordlists=self.list_of_wordlists)
                dic[theme_shares_obj.id] = shares_list
            df = pd.DataFrame(dic).T
            df.columns = [wordlist[0] for wordlist in self.list_of_wordlists]
            self.data_matrix_df = df

