import pickle
import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from preprocessing.text import Text
from preprocessing.SNA import CharacterNetwork
import networkx as nx
from copy import deepcopy
from collections import Counter


"""
2. Generate word lists from corpus vocabulary pased on POS-Tagging (with spaCy), for example: function word lists, semantic word lists, noun lists, verb lists, adjective lists etc.
"""

''' 3. Generating, editing, manipulating, and transforming a Corpus Representation as term document matrix:  
The following scripts organize the preprocessing steps of:
1. a) Correcting common and domain specific OCR errors, based on practical experience with reading OCR; and b) Normalization if necessary (transforming Ä,Ö,Ü,ß to ae, oe, ue, ss for example for topic modeling with mallet
2. Lemmatization (with spacy)
3. POS-Tagging, (with spacy)
4. Selection of POS-Types
5. Stopword-Handling
6. Chunking
'''

# Achtung, diese Klasse muss wieder für mallet outpout angepasst werden!
class DocFeatureMatrix():
    """
    abstract parent class for TDM (= term document matrix) and TopicDocMatrix as child classes. Feature vectors for each document are represented as row vectors.
    """
    def __init__(self, data_matrix_filepath, metadata_csv_filepath=None, data_matrix_df=None, metadata_df=None, mallet=False):
        self.data_matrix_df = data_matrix_df
        self.data_matrix_filepath = data_matrix_filepath
        self.metadata_csv_filepath = metadata_csv_filepath
        self.metadata_df = metadata_df
        self.mallet = mallet

        print("data_csv_filepath is:", self.data_matrix_filepath)
        print("metadata_csv_filepath is:", self.metadata_csv_filepath)


        if self.metadata_csv_filepath is not None:
            self.metadata_df = pd.read_csv(self.metadata_csv_filepath, index_col=0)

        if self.data_matrix_filepath is not None:

            if self.mallet == False:
                self.data_matrix_df = pd.read_csv(self.data_matrix_filepath, index_col=0)

            elif self.mallet == True:
                print( "hier muss die Anpassung an mallet output df erfolgen: an doc-topic-matrix")
                df = pd.read_csv(self.data_matrix_filepath, index_col=1, sep='\t', header=None)
                df.drop(df.columns[0], axis=1, inplace=True)
                #df.columns = [i for i in range(len(df.columns))]
                df.reset_index(inplace=True)
                self.data_matrix_df = df



    def reduce_to(self, reduction_list, return_eliminated_terms_list=False):
        """
        Based on the vocabulary of matrix_df, a new FeatureDoc_matrix object is returned only with features which are listed in reduction_list.
        if flag return_eliminated_terms_list == True, also the list of eliminated terms is returned
        if flag return_eliminated_terms_list == True, only a reduced DTM instance is returned.
        """
        object = deepcopy(self)
        columns_list = list(object.data_matrix_df.columns.values)
        terms_to_reduce = list(filter(lambda x: x in reduction_list, columns_list))
        eliminated_terms_list = list(filter(lambda x: x not in reduction_list, columns_list))
        object.data_matrix_df = object.data_matrix_df.loc[:, terms_to_reduce]
        if return_eliminated_terms_list == False:
            return object
        else:
           return object,  eliminated_terms_list

    def eliminate(self, elimination_list):
        """
        Based on the vocabulary of matrix_df, a new FeatureDoc_matrix object is returned. All
        features listed in elimination list are eliminated from the data_matrix_df.
        Typical use cases are stopword lists and name lists.

        """
        object = deepcopy(self)
        columns_list = list(object.data_matrix_df.columns.values)
        terms_to_drop = list(filter(lambda x: x in elimination_list, columns_list))
        object.data_matrix_df.drop(terms_to_drop, axis=1, inplace=True)

        return object


    def add_metadata(self, metadata_category):
        """
        adds metadata from metadata table to the data_matrix_df attribute in new object and overwrites this attribute with the enriched matrix.
        :param metadata_category: Type of metadata as specified as column name in the meta data table (from csv)
        :return: a copy of the object with data_matrix_df attribute is generated. This attribute is a data frame and supplied with the respective metadata category as an additional column
        """
        object = deepcopy(self)
        object.data_matrix_df = object.data_matrix_df.join(self.metadata_df[metadata_category])
        return object

    def reduce_to_categories(self, metadata_category, label_list):
        values_dict = {metadata_category: label_list}
        object = deepcopy(self)
        object.data_matrix_df = object.data_matrix_df[object.data_matrix_df.isin(values_dict).any(1)]
        return object

    def save_csv(self, outfile_path):
        """
        save feature document matrix in the data_matrix_df attribute as csv file under outfile_path
        :param outfile_path: filepath to store the file.
        """
        self.data_matrix_df.to_csv(path_or_buf=outfile_path)



class DTM(DocFeatureMatrix):
    """
    The DTM (document term matrix) class, which is a sub class of DocFeatureMatrix serves to convert textual data stored as several txt files to a document term matrix. The word vectors for each document are represented as row vectors.
    It inherits the methods of its parent class to load, supply, edit and manipulate term document matrices, and to add meta data from metadata csv file.

    The procedure of assigning texts to metadata is based on numerical document IDs with 7 digits (XXXXX-XX), the ID is stored in the filename of each
    text file and as index in the metadata table.

    Attributes:
       The term document matrix can be loaded with .generate() to the .corpus_df attribute as pandas DataFrame from variable corpus_df, metadata_csv_filepath, or a corpus with texts in corpus_path
       Additionally, the metadata table will be loaded to metatda_df attribute as pandas DataFrame with .generate()


    :param
    normalization: the type of normalization to be used:
           "abs": absolute token counts of term per document
           "l1": relative frequency: sum of tokens of a word type per document.
           "l2": quadratic l2-normalization.
           "tf_idf": term frequency inverse document frequency weighting.
    lemmatize (bool; default = False): If True, lemmatized tokens are counted; if False, unlemmatized tokens are counted.
    dehyphen (bool; default = False): If True, hyphenation in the text files is removed.
    n_mfw: number of most frequent words to be counted, if set to 0, all words are counted.
    """

    def __init__(self, corpus_path=None, data_matrix_df=None, data_matrix_filepath=None, metadata_csv_filepath=None,
                 normalize_orthogr=False, normalization_table_path=None,
                 metadata_df=None, encoding="utf-8", normalization="tfidf", correct_ocr=True, eliminate_pagecounts=True, handle_special_characters=True,
                 inverse_translate_umlaute=False, eliminate_pos_items=True, list_of_pos_tags=None, lemmatize=True, remove_hyphen=True, sz_to_ss=False, translate_umlaute=False,
                 remove_stopwords=False, stoplist_filepath=None, n_mfw=0, mallet=False, language_model=None, **kwargs):
        DocFeatureMatrix.__init__(self, data_matrix_df, data_matrix_filepath, metadata_csv_filepath, metadata_df, mallet)
        self.corpus_path = corpus_path
        self.handle_special_characters = handle_special_characters
        self.encoding = encoding
        self.normalization = normalization
        self.correct_ocr = correct_ocr
        self.handle_special_characters = handle_special_characters
        self.inverse_translate_umlaute = inverse_translate_umlaute
        self.eliminate_pagecounts = eliminate_pagecounts
        self.eliminate_pos_itmes = eliminate_pos_items
        self.list_of_pos_tags = list_of_pos_tags
        self.lemmatize = lemmatize
        self.sz_to_ss = sz_to_ss
        self.translate_umlaute = translate_umlaute
        self.remove_hyphen = remove_hyphen
        self.remove_stopwords = remove_stopwords
        self.encoding = encoding
        self.language_model = language_model
        self.normalize_orthogr = normalize_orthogr
        self.normalization_table_path = normalization_table_path
        self.stoplist_filepath = stoplist_filepath
        self.n_mfw = n_mfw

    def _corpus_as_dict(self):
        """
        private method which calls Text class to preprocess text over all texts in corpus path and to store the processed text
         as value with the document id as key in a dictionary.
        This method is to be called by generate_from_textcorpus method as the basis for vectorization.
        retunrs: dictionary with doc ids as keys and processed text for each document as values.
        """
        dic = {}
        for filepath in os.listdir(self.corpus_path):
            text_object = Text(filepath=os.path.join(self.corpus_path, filepath), remove_hyphen=self.remove_hyphen, correct_ocr=self.correct_ocr,
                                normalize_orthogr=self.normalize_orthogr, normalization_table_path=self.normalization_table_path,
                                handle_special_characters=self.handle_special_characters,
                                inverse_translate_umlaute=self.inverse_translate_umlaute,
                                lemmatize=self.lemmatize, sz_to_ss=self.sz_to_ss,
                                translate_umlaute=self.translate_umlaute, language_model=self.language_model)
            text_object()
            print("currently processes text with id: "+str(text_object.id))
            dic[text_object.id] = text_object.text
        return dic




    def generate_from_textcorpus(self):
        """
        This function generates or loads the metadata table and the term document matrix with the parameters specified in the CORPUS_DTM instance
        :return: attributes self.metadata_df with metadata table and self.corpus_df with term document matrix as DataFrame.
        The dictionary hast the form:  {'00000-00': 'This is the text of doc0', '00001-00' : 'This is text of doc1.'}
        """

        if self.metadata_df == None and self.metadata_csv_filepath is not None:
            self.load_metadata_file()
        else:
           pass

        if self.corpus_path is None and self.data_matrix_filepath is not None:
            self.load_data_matrix_file()
        elif self.corpus_path is None and self.data_matrix_filepath is None:
            if self.data_matrix_df is not None:
                self.data_matrix_df = self.data_matrix_df
        elif self.corpus_path is not None and self.data_matrix_df == None and self.data_matrix_filepath == None:
            # create a term document matrix as pandas DataFrame with private _corpus_as_dict() method:
            corpus_dict = self._corpus_as_dict()
            list_texts = corpus_dict.values()
            list_docids = corpus_dict.keys()
            vectorizer = CountVectorizer()
            fit_all = vectorizer.fit(list_texts)
            matrix_all = vectorizer.fit_transform(list_texts)

            if (self.normalization == "l1" or self.normalization == "l2"):
                freq_matrix = normalize(matrix_all, norm=self.normalization)

            elif self.normalization == "tfidf":
                vectorizer = TfidfVectorizer()
                freq_matrix = vectorizer.fit_transform(list_texts)

            elif self.normalization == "abs":
                freq_matrix = matrix_all.copy()

            else:
                # use the matrix with absolute token counts as fall back level:
                freq_matrix = matrix_all.copy()

            if self.n_mfw == 0:
                freq_array = freq_matrix.toarray()
                self.data_matrix_df = pd.DataFrame(freq_array, index=list_docids,
                              columns=vectorizer.get_feature_names())
            elif self.n_mfw != 0:
                sum_words_vector_all = matrix_all.sum(axis=0)
                list_words_freq = sorted([(word, ID, sum_words_vector_all[0, ID]) for word, ID
                                      in fit_all.vocabulary_.items()],
                                     key=lambda x: x[2], reverse=True)
                list_mfw_ids = [item[1] for item in list_words_freq[:self.n_mfw]]
                list_mfw = [item[0] for item in list_words_freq[:self.n_mfw]]
                freq_array = freq_matrix.toarray()[:,list_mfw_ids]

                self.data_matrix_df = pd.DataFrame(freq_array, index=list_docids,
                          columns=list_mfw)







class Junk_Corpus():
    def __init__(self, corpus_path=None, outfile_directory=None, correct_ocr=True, eliminate_pagecounts=True,
                 handle_special_characters=True, inverse_translate_umlaute=False, lemmatize=True,
                 remove_hyphen=True, sz_to_ss=False, translate_umlaute=False,
                 eliminate_pos_items=True, list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"],
                 keep_pos_items=False, list_keep_pos_tags=None,
                 segmentation_type="fixed", fixed_chunk_length=600, num_chunks=5,
                 stopword_list=None, remove_stopwords="after_chunking", language_model=None, **kwargs):
        self.corpus_path = corpus_path
        self.outfile_directory = outfile_directory
        self.correct_ocr = correct_ocr
        self.eliminate_pagecounts = eliminate_pagecounts
        self.handle_special_characters = handle_special_characters
        self.inverse_translate_umlaute = inverse_translate_umlaute
        self.lemmatize = lemmatize
        self.remove_hyphen =remove_hyphen
        self.sz_to_ss = sz_to_ss
        self.translate_umlaute = translate_umlaute
        self.remove_stopwords = remove_stopwords
        self.segmentation_type = segmentation_type
        self.fixed_chunk_length = fixed_chunk_length
        self.num_chunks = num_chunks
        self.eliminate_pos_items = eliminate_pos_items
        self.keep_pos_items = keep_pos_items
        self.list_keep_pos_tags = list_keep_pos_tags
        self.list_eliminate_pos_tags = list_eliminate_pos_tags
        self.stopword_list = stopword_list
        self.language_model = language_model

    def __call__(self):
        for filename in os.listdir(self.corpus_path):
            text_object = Text(filepath=os.path.join(self.corpus_path, filename), correct_ocr=self.correct_ocr,
                               eliminate_pagecounts=self.eliminate_pagecounts,
                               handle_special_characters=self.handle_special_characters,
                               inverse_translate_umlaute=self.inverse_translate_umlaute,
                               lemmatize=self.lemmatize, remove_hyphen=self.remove_hyphen, sz_to_ss=self.sz_to_ss,
                               translate_umlaute=self.translate_umlaute,
                               eliminate_pos_items=self.eliminate_pos_items, list_eliminate_pos_tags=self.list_eliminate_pos_tags,
                               keep_pos_items=self.keep_pos_items, list_keep_pos_tags=self.list_keep_pos_tags,
                               remove_stopwords=self.remove_stopwords, stopword_list=self.stopword_list,
                               language_model=self.language_model)
            text_object()
            text_object.f_chunking(segmentation_type=self.segmentation_type, fixed_chunk_length=self.fixed_chunk_length, num_chunks=self.num_chunks)
            text_object.f_save_chunks(self.outfile_directory)


def generate_text_files(chunking=False, pos_representation=False, corpus_path=None, outfile_path=None,
                        only_selected_files=False, list_of_file_ids=None, language_model=None,
                        correct_ocr=True, eliminate_pagecounts=True,
                 handle_special_characters=True, inverse_translate_umlaute=False, lemmatize=False,
                 remove_hyphen=True, sz_to_ss=False, translate_umlaute=False,
                 eliminate_pos_items=False, list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"],
                 keep_pos_items=False, list_keep_pos_tags=None,
                 segmentation_type="fixed", fixed_chunk_length=600, num_chunks=5,
                        normalize_orthogr=False, normalization_table_path=None,
                 stopword_list=None, remove_stopwords=None):
    """
    generate text files from corpus files, such as chunks, a selection of files based on metadata selection, or POS-/NER-Representation
    :param chunking: Bool.: If "True", the chunking pipeline is activated, if "False",
    :param pos_representation: Bool.: If "True", plain text files with text id as filename are generated, where each line consists of string, POS-Tag,
        and, eventually, NER entity Type for each token in text.
    :param corpus_path:
    :param outfile_path:
    :param list_of_file_ids:
    :param language_model:
    :param correct_ocr:
    :param eliminate_pagecounts:
    :param handle_special_characters:
    :param inverse_translate_umlaute:
    :param lemmatize:
    :param remove_hyphen:
    :param sz_to_ss:
    :param translate_umlaute:
    :param eliminate_pos_items:
    :param list_eliminate_pos_tags:
    :param keep_pos_items:
    :param list_keep_pos_tags:
    :param segmentation_type:
    :param fixed_chunk_length:
    :param num_chunks:
    :param stopword_list:
    :param remove_stopwords:
    :return:
    """
    if not os.path.exists(outfile_path):
        os.makedirs(outfile_path)
    for filename in os.listdir(corpus_path):
        text_obj = Text(filepath=os.path.join(corpus_path, filename), correct_ocr=correct_ocr,
                               eliminate_pagecounts=eliminate_pagecounts,
                               handle_special_characters=handle_special_characters,
                               inverse_translate_umlaute=inverse_translate_umlaute,
                               lemmatize=lemmatize, remove_hyphen=remove_hyphen, sz_to_ss=sz_to_ss,
                               translate_umlaute=translate_umlaute,
                               eliminate_pos_items=eliminate_pos_items, list_eliminate_pos_tags=list_eliminate_pos_tags,
                               keep_pos_items=keep_pos_items, list_keep_pos_tags=list_keep_pos_tags,
                               remove_stopwords=remove_stopwords, stopword_list=stopword_list,
                        normalize_orthogr=normalize_orthogr, normalization_table_path=normalization_table_path,
                        language_model=language_model)
        text_obj.f_extract_id()

        if only_selected_files == True:

            if text_obj.id in list_of_file_ids:
                print("currently processes file with id: "+ str(text_obj.id))
                text_obj.f_read_file()
                if remove_hyphen == True:
                    text_obj.f_remove_hyphen()
                if correct_ocr == True:
                    text_obj.f_correct_ocr()
                if eliminate_pagecounts == True:
                    text_obj.f_eliminate_pagecounts()
                if handle_special_characters == True:
                    text_obj.f_handle_special_characters()
                if inverse_translate_umlaute == True:
                    text_obj.f_inverse_translate_umlaute()
                if sz_to_ss == True:
                    text_obj.f_sz_to_ss()
                if translate_umlaute == True:
                    text_obj.f_translate_umlaute()

                if normalize_orthogr == True:
                    text_obj.f_normalize_orthogr()
                if lemmatize == True:
                    text_obj.f_generate_pos_triples()
                    text_obj.f_lemmatize()

                if chunking == False:
                    if pos_representation == False:
                        text_obj.f_save_text(outfile_path)
                    elif pos_representation == True:
                        text_obj.f_check_save_pos_ner_parsing(outfile_path)
                elif chunking == True:
                    if pos_representation == True:
                        print("Warning: POS-Representation is not implemented for operation on junks. Use chunking == False!")
                        pass
                    elif pos_representation == False:
                        text_obj.f_chunking(segmentation_type=segmentation_type,
                                       fixed_chunk_length=fixed_chunk_length, num_chunks=num_chunks)
                        text_obj.f_save_chunks(outfile_path)

            else:
                pass

        elif only_selected_files == False:
            print("currently processes file with id: " + str(text_obj.id))

            text_obj.f_read_file()
            if remove_hyphen == True:
                text_obj.f_remove_hyphen()
            if correct_ocr == True:
                text_obj.f_correct_ocr()
            if eliminate_pagecounts == True:
                text_obj.f_eliminate_pagecounts()
            if handle_special_characters == True:
                text_obj.f_handle_special_characters()
            if inverse_translate_umlaute == True:
                text_obj.f_inverse_translate_umlaute()
            if sz_to_ss == True:
                text_obj.f_sz_to_ss()
            if translate_umlaute == True:
                text_obj.f_translate_umlaute()

            if normalize_orthogr == True:
                text_obj.f_normalize_orthogr()
            if lemmatize == True:
                text_obj.f_generate_pos_triples()
                text_obj.f_lemmatize()

            if chunking == False:
                if pos_representation == False:
                    text_obj.f_save_text(outfile_path)
                elif pos_representation == True:
                    text_obj.f_check_save_pos_ner_parsing(outfile_path)
            elif chunking == True:
                if pos_representation == True:
                    print(
                        "Warning: POS-Representation is not implemented for operation on junks. Use chunking == False!")
                    pass
                elif pos_representation == False:
                    text_obj.f_chunking(segmentation_type=segmentation_type,
                                        fixed_chunk_length=fixed_chunk_length, num_chunks=num_chunks)
                    text_obj.f_save_chunks(outfile_path)

            if chunking == False:
                if pos_representation == False:
                    text_obj.f_save_text(outfile_path)
                elif pos_representation == True:
                    text_obj.f_check_save_pos_ner_parsing(outfile_path)
            elif chunking == True:
                if pos_representation == True:
                    print(
                        "Warning: POS-Representation is not implemented for operation on junks. Use chunking == False!")
                    pass
                elif pos_representation == False:
                    text_obj.f_chunking(segmentation_type=segmentation_type,
                                        fixed_chunk_length=fixed_chunk_length, num_chunks=num_chunks)
                    text_obj.f_save_chunks(outfile_path)

            else:
                pass

def check_save_pos_ner_parsing_corpus(corpus_path, outfile_path, list_of_file_ids, language_model, correct_ocr=True, eliminate_pagecounts=True,
                 handle_special_characters=True, inverse_translate_umlaute=False, lemmatize=True,
                 remove_hyphen=True, sz_to_ss=False, translate_umlaute=False,
                 eliminate_pos_items=True, list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"],
                 keep_pos_items=False, list_keep_pos_tags=None,
                 segmentation_type=None, fixed_chunk_length=None, num_chunks=None,
                 stopword_list=None, remove_stopwords=None):
    for filename in os.listdir(corpus_path):
        text_obj = Text(filepath=os.path.join(corpus_path, filename), correct_ocr=correct_ocr,
                               eliminate_pagecounts=eliminate_pagecounts,
                               handle_special_characters=handle_special_characters,
                               inverse_translate_umlaute=inverse_translate_umlaute,
                               lemmatize=lemmatize, remove_hyphen=remove_hyphen, sz_to_ss=sz_to_ss,
                               translate_umlaute=translate_umlaute,
                               eliminate_pos_items=eliminate_pos_items, list_eliminate_pos_tags=list_eliminate_pos_tags,
                               keep_pos_items=keep_pos_items, list_keep_pos_tags=list_keep_pos_tags,
                               remove_stopwords=remove_stopwords, stopword_list=stopword_list, language_model=language_model)
        text_obj.f_extract_id()
        if text_obj.id in list_of_file_ids:
            text_obj()

        else:
            pass


class DocNetworkfeature_Matrix(DocFeatureMatrix):
    def __init__(self, corpus_path=None, data_matrix_df=None, data_matrix_filepath=None, metadata_csv_filepath=None,
                 metadata_df=None, encoding="utf-8", normalization="tfidf", correct_ocr=True, eliminate_pagecounts=True, handle_special_characters=True,
                 inverse_translate_umlaute=False, eliminate_pos_items=False, list_of_pos_tags=None, lemmatize=True, remove_hyphen=True, sz_to_ss=False, translate_umlaute=False,
                 remove_stopwords=False, stoplist_filepath=None, n_mfw=0, segmentation_type="paragraph", fixed_chunk_length=1000, num_chunks=5, language_model=None, mallet=False,
                 corpus_as_dict=None, corpus_characters_list=None, **kwargs):
        DocFeatureMatrix.__init__(self, data_matrix_df, data_matrix_filepath, metadata_csv_filepath, metadata_df, mallet)
        self.corpus_path = corpus_path
        self.handle_special_characters = handle_special_characters
        self.encoding = encoding
        self.normalization = normalization
        self.correct_ocr = correct_ocr
        self.handle_special_characters = handle_special_characters
        self.inverse_translate_umlaute = inverse_translate_umlaute
        self.eliminate_pagecounts = eliminate_pagecounts
        self.eliminate_pos_items = eliminate_pos_items
        self.list_of_pos_tags = list_of_pos_tags
        self.lemmatize = lemmatize
        self.sz_to_ss = sz_to_ss
        self.translate_umlaute = translate_umlaute
        self.remove_hyphen = remove_hyphen
        self.remove_stopwords = remove_stopwords
        self.encoding = encoding
        self.segmentation_type = segmentation_type
        self.fixed_chunk_length = fixed_chunk_length
        self.num_chunks = num_chunks
        self.language_model = language_model
        self.stoplist_filepath = stoplist_filepath
        self.n_mfw = n_mfw
        self.corpus_as_dict = corpus_as_dict
        self.corpus_characters_list = corpus_characters_list

        """
        call Text class to preprocess text over all texts in corpus path and to store the processed text
         as value with the document id as key in a dictionary.
        This method is to be called by generate_from_textcorpus method as the basis for vectorization.
        returns dictionary with doc ids as keys and processed text for each document as values.
        """
        if self.corpus_as_dict is None:
            dic = {}
            corpus_characters_list = []
            for filepath in os.listdir(self.corpus_path):
                char_netw = CharacterNetwork(filepath=os.path.join(self.corpus_path, filepath), minimal_reference=2, text=None, id=None, chunks=None,
                                         pos_triples=None, remove_hyphen=True,
                                         correct_ocr=self.correct_ocr, eliminate_pagecounts=self.eliminate_pagecounts, handle_special_characters=self.handle_special_characters,
                                         inverse_translate_umlaute=self.inverse_translate_umlaute,
                                         eliminate_pos_items=self.eliminate_pos_items, list_keep_pos_tags=self.list_of_pos_tags,
                                         list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"], lemmatize=False,
                                         sz_to_ss=False, translate_umlaute=False, max_length=5000000,
                                         remove_stopwords="before_chunking", stopword_list=None,
                                         language_model=self.language_model)
                char_netw()
                char_netw.f_chunking(segmentation_type=self.segmentation_type, fixed_chunk_length=self.fixed_chunk_length, num_chunks=self.num_chunks)
                char_netw.generate_characters_graph()

                dic[char_netw.id] = [". ".join(char_netw.characters_list), len(char_netw.characters_list), nx.density(char_netw.graph), char_netw.proportion_of_characters_with_degree(value_degree_centrality=1), char_netw.token_length]
                corpus_characters_list += char_netw.characters_list
            self.corpus_as_dict = dic
            self.corpus_characters_list = corpus_characters_list

    def generate_df(self):
        df = pd.DataFrame(self.corpus_as_dict).T
        df.columns = ["Figuren", "Figurenanzahl", "Netwerkdichte", "Anteil Figuren mit degree centrality == 1", "Länge in Token"]
        self.data_matrix_df = df

    def corpus_characters_list_to_file(self, outfilepath):
        names_string = ", ".join(map(str, set(self.corpus_characters_list)))
        with open(outfilepath, "w") as infile:
            infile.write(names_string.replace(", ", "\n"))
        pass


    def corpus_character_counter(self):
        return Counter(self.corpus_characters_list)

class POS_Vocab():
    """
        :param corpus_path:
        :return: Three lists: List with all verb types, all noun types and all adjective types in corpus
    """
    def __init__(self, token_pos_dict=None, lemma_pos_dict=None, corpus_path=None, nouns=None, verbs=None, adj=None, adv=None, propn=None, intj=None, function_words=None, other=None,
                 max_length=5000000, language_model = None):
        self.token_pos_dict = token_pos_dict
        self.lemma_pos_dict = lemma_pos_dict
        self.corpus_path = corpus_path
        self.nouns = nouns
        self.verbs = verbs
        self.adj = adj
        self.adv = adv
        self.propn = propn
        self.intj = intj
        self.function_words = function_words
        self.other = other
        self.language_model = language_model
        self.max_length = max_length

    def generate_pos_lists(self, lemma=True, token=True):
        token_noun_list = []
        lemma_noun_list = []
        token_verb_list = []
        lemma_verb_list =[]
        token_adj_list = []
        lemma_adj_list = []
        token_adv_list = []
        lemma_adv_list = []
        token_intj_list =[]
        lemma_intj_list = []
        token_propn_list = []
        lemma_propn_list = []
        other_list = []
        function_word_list =[]
        lemma_pos_dict = {}
        token_pos_dict = {}
        for filename in os.listdir(self.corpus_path):
            raw_text = open(os.path.join(self.corpus_path, filename), 'r').read()
            nlp = spacy.load(self.language_model)
            nlp.max_length = self.max_length
            doc = nlp(self.text[:self.max_length], disable='ner')
            for token in doc:
                if token.pos_ == "NOUN":
                    token_noun_list.append(token.text.lower())
                    lemma_noun_list.append(token.lemma_.lower())
                elif token.pos_ == "VERB":
                    token_verb_list.append(token.text.lower())
                    lemma_verb_list.append(token.lemma_.lower())
                elif token.pos_ == "ADJ":
                    token_adj_list.append(token.text.lower())
                    lemma_adj_list.append(token.lemma_.lower())
                elif token.pos_ == "ADV":
                    token_adv_list.append(token.text.lower())
                    lemma_adv_list.append(token.lemma_.lower())
                elif token.pos == "INTJ":
                    token_intj_list.append(token.text.lower())
                    lemma_intj_list.append(token.lemma_lower())
                elif token.pos_ == "PROPN":
                    token_propn_list.append(token.text.lower())
                    lemma_propn_list.append(token.lemma_.lower())
                elif token.pos_ in ["AUX", "CONJ", "ADP", "CCONJ", "DET", "PART", "PRON", "SCONJ"]:
                    function_word_list.append(token.text)
                elif token.pos in ["X", "SYM", "PUNCT", "SPACE", "NUM"]:
                    other_list.append(token.text.lower())
                else:
                    other_list.append(token.text.lower())

        token_pos_dict["nouns"] = ", ".join(map(str, set(token_noun_list)))
        token_pos_dict["verbs"] = ", ".join(map(str, set(token_verb_list)))
        token_pos_dict["adj"] = ", ".join(map(str, set(token_adj_list)))
        token_pos_dict["adv"] = ", ".join(map(str, set(token_adv_list)))
        token_pos_dict["intj"] = ", ".join(map(str, set(token_intj_list)))
        token_pos_dict["function_words"] = ", ".join(map(str, set(function_word_list)))
        token_pos_dict["propn"] = ", ".join(map(str, set(token_propn_list)))
        token_pos_dict["other"] = ", ".join(map(str, set(other_list)))
        lemma_pos_dict["nouns"] = ", ".join(map(str, set(lemma_noun_list)))
        lemma_pos_dict["verbs"] = ", ".join(map(str, set(lemma_verb_list)))
        lemma_pos_dict["adj"] = ", ".join(map(str, set(lemma_adj_list)))
        lemma_pos_dict["adv"] = ", ".join(map(str, set(lemma_adv_list)))
        if not lemma_intj_list:
            lemma_pos_dict["intj"] = ", "
        elif lemma_intj_list:
            lemma_pos_dict["intj"] = ", ".join(map(str, set(token_intj_list)))
        lemma_pos_dict["function_words"] = ", ".join(map(str, set(function_word_list)))
        lemma_pos_dict["propn"] = ", ".join(map(str, set(lemma_propn_list)))
        lemma_pos_dict["other"] = ", ".join(map(str, set(other_list)))
        self.token_pos_dict = token_pos_dict
        self.lemma_pos_dict = lemma_pos_dict


    def save_all(self, directory, lemmatized=True, non_lemmatized=True):
        if lemmatized == True:
            with open (os.path.join(directory, "nouns_lemma_list_vocab.txt"), "w") as infile:
                infile.write(self.lemma_pos_dict["nouns"].replace(", ", "\n"))
            with open(os.path.join(directory, "verbs_lemma_list_vocab.txt"), "w") as infile:
                infile.write(self.lemma_pos_dict["verbs"].replace(", ", "\n"))
            with open(os.path.join(directory, "adj_lemma_list_vocab.txt"), "w") as infile:
                infile.write(self.lemma_pos_dict["adj"].replace(", ", "\n"))
            with open(os.path.join(directory, "adv_lemma_list_vocab.txt"), "w") as infile:
                infile.write(self.lemma_pos_dict["adv"].replace(", ", "\n"))
            with open(os.path.join(directory, "propn_lemma_list_vocab.txt"), "w") as infile:
                infile.write(self.lemma_pos_dict["propn"].replace(", ", "\n") + "\n")
            with open(os.path.join(directory, "function_word_raw_list_corpus_vocab.txt"), "w") as infile:
                infile.write(self.lemma_pos_dict["function_words"].replace(", ", "\n"))
            with open(os.path.join(directory, "pos_dict_lemma_vocab.pickle"), "wb") as handle:
                pickle.dump(self.lemma_pos_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pass

        if non_lemmatized == True:
            with open (os.path.join(directory, "nouns_raw_list_vocab.txt"), "w") as infile:
                infile.write(self.token_pos_dict["nouns"].replace(", ", "\n"))
            with open(os.path.join(directory, "verbs_raw_list_vocab.txt"), "w") as infile:
                infile.write(self.token_pos_dict["verbs"].replace(", ", "\n"))
            with open(os.path.join(directory, "adj_raw_list_vocab.txt"), "w") as infile:
                infile.write(self.token_pos_dict["adj"].replace(", ", "\n"))
            with open(os.path.join(directory, "adv_raw_list_vocab.txt"), "w") as infile:
                infile.write(self.token_pos_dict["adv"].replace(", ", "\n"))
            with open(os.path.join(directory, "propn_raw_list_vocab.txt"), "w") as infile:
                infile.write(self.token_pos_dict["propn"].replace(", ", "\n") + "\n")
            with open(os.path.join(directory, "function_words_raw_list_corpus_vocab.txt"), "w") as infile:
                infile.write(self.token_pos_dict["function_words"].replace(", ", "\n"))
            with open(os.path.join(directory, "pos_dict_raw_vocab.pickle"), "wb") as handle:
                pickle.dump(self.token_pos_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pass
