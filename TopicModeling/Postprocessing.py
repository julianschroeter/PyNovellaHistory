
import re
import pandas as pd
from collections import Counter
from preprocessing.corpus import DocFeatureMatrix
import spacy
from copy import deepcopy


# Achtung! Diese Klasse ist im Moment korrupt, hier ist eine frühere Version irrtümlich wieder eingespielt. Sie passt nicht zu FeatureDocMatrix
# Diese Klasse sollte wieder angepasst werden.
class DocTopicMatrix(DocFeatureMatrix):
    """
    Child class of FeatureDocMatrix. Creates objects with topic-document-matrix and meta data table as attributes. This class provides methods for loading and operating with the matrix.
    The matrix in the .data_matrix_attribute has a multi index: On the first level, there are the ids of the document (form: 00000-00),
    on the second level, there are the chunks (0000) of each document.
    """
    def __init__(self, data_matrix_filepath,data_matrix_df, metadata_csv_filepath, metadata_df, mallet):
        print("data_csv_filepath ist", data_matrix_filepath)
        print("metadata_csv_filepath ist", metadata_csv_filepath)
        super().__init__(data_matrix_filepath,data_matrix_df, metadata_csv_filepath, metadata_df, mallet)


    def adjust_doc_chunk_multiindex(self):
        self.data_matrix_df["doc_id"] = self.data_matrix_df.iloc[:, 0].apply(lambda x: re.search("\d{5}-\d{2}", x).group() if re.search("\d{5}-\d{2}", x) else x)
        self.data_matrix_df["chunk_count"] = self.data_matrix_df.iloc[:, 0].apply(lambda x: re.search("(?<=_)\d{4}", x).group() if re.search("_\d{4}", x) else 0)  # with positive lookbehind (?<=...)
        self.data_matrix_df = self.data_matrix_df.set_index(["doc_id", "chunk_count"])
        self.data_matrix_df = self.data_matrix_df.drop(columns=[1])
        self.data_matrix_df.columns = range(self.data_matrix_df.shape[1])
        self.data_matrix_df.sort_index(level=["doc_id", "chunk_count"], sort_remaining=False)
        return self


    def mean_doclevel(self):
        """
        calculates the average topic distributions over all chunks per document, based on data_matrix_df
        returns a manipulated copy of TopicDocMatrix where data_matrix_df is reduced to the mean of topic distribution for each document over all chunks
        """
        object = deepcopy(self)
        object.data_matrix_df = object.data_matrix_df.groupby(level=0).mean()
        object.data_matrix_df.index.name = None
        return object

    def last_chunk(self):
        """
        returns a manipulated copy of FeatureDocMatrix where data_matrix_df is reduced to the last chunk per document.
        Use case: exploring topics at the end of texts, for example in order to detect happy or tragic endings.
        """
        object = deepcopy(self)
        object.data_matrix_df = object.data_matrix_df.groupby(level=0).last()
        object.data_matrix_df.index.name = None
        return object

    def first_chunk(self):
        """
        returns processing_df as reduced to the first chunk per document.
        Use case: exploring topics at the beginning of texts
        """
        object = deepcopy(self)
        object.data_matrix_df.groupby(level=0).first()
        object.data_matrix_df.index.name = None
        return object




class TopicKeys(pd.DataFrame):
    def __init__(self, infilepath=None):
        super().__init__(pd.read_csv(filepath_or_buffer=infilepath, index_col=0, sep="\t", names=["topic_id", "weight", "keys"]))

    def pos_counts(self, pos_type="VERB", count=6, language_model=None):
        nlp = spacy.load(language_model)
        self["counter_POS"] = self["keys"].apply(lambda x: Counter([token.pos_ for token in nlp(x)]))
        self["to_drop"] = self["counter_POS"].apply(lambda x: True if x[pos_type] > count else False)

    def list_topics_to_drop(self):
        return self.index[self["to_drop"]].tolist()
