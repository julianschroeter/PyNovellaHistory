import os
from collections import Counter
import numpy as np
import spacy



'''
1. Functions to set global working directories depending on different personal computer systems.
'''


def set_DistReading_directory(system_name):
    """
       :param system_name: "my_mac", "my_xps", or "wcph104". Here, the respective Computer system with its respective directory structure has to be selected"
       :return: the path for the the DistReading directory in Spaces (common cloud for all systems)
    """
    if system_name == "my_mac":
        return "/Users/karolineschroter/Spaces/DistReading"
    elif system_name == "my_xps":
        return "/home/julian/Spaces/DistReading"
    elif system_name == "wcph104":
        return os.path.join( "C:" + os.sep, "Users", "jus71hy", "Documents", "Spaces", "DistReading")
        pass
    else:
        print("Warning: The file system has not been specified correctly! Filepath is set to my_xps per default")
        return "/home/julian/Spaces/DistReading"


def set_system_data_directory(system_name):
    """
    :param system_name: "my_mac", "my_xps", "wcph113" (which is a remote server) or "wcph104". Here, the respective Computer system with its respective directory structure has to be selected"
    :return: the path for the the working project data
    """
    if system_name in ["wcph104", "my_mac", "my_xps"]:
        return os.path.join(set_DistReading_directory(system_name), "data")
        pass
    elif system_name == "wcph113":
        return "/mnt/data/users/schroeter"
        pass



def local_temp_directory(system_name):
    if system_name == "my_mac":
        return "/Users/karolineschroter/Documents/CLS_temp"
    elif system_name == "my_xps":
        return "/home/julian/Documents/CLS_temp"
    elif system_name == "wcph104":
        return os.path.join("C:" + os.sep, "Users", "jus71hy", "Documents", "CLS_temp")
    elif system_name == "wcph113":
        return "/mnt/data/users/schroeter/CLS_temp"

def global_corpus_directory(system_name, test=False):
    """
    :param system_name: Here, the respective Computer system with its respective directory structure has to be selected"
    "test": If True, the directory for a samm test sample is selected; if False: the whole corpus is selected.
    :return: the path for the directory with all plain text files of the project corpus for the system specified with the parameter
    """
    if test == False:
            return os.path.join(set_system_data_directory(system_name), "novella_corpus_all")
    elif test == True:
            return os.path.join(set_system_data_directory(system_name), "novella_corpus_test")

def global_corpus_representation_directory(system_name):
    """
        :param system_name: Here, the respective Computer system with its respective directory structure has to be selected"
        :return: the path for the directory to store all corpus representation files such as dtms or lists for the system specified with the parameter
        """
    if system_name == "wcph113":
        return os.path.join(set_system_data_directory(system_name), "novella_corpus_representation")
    else:
        return os.path.join(set_system_data_directory(system_name), "novella_corpus_representation")


def global_corpus_raw_dtm_directory(system_name):
    """
        :param system_name: Here, the respective Computer system with its respective directory structure has to be selected"
        :return: the path for the directory to store all corpus representation files such as dtms or lists for the system specified with the parameter
        """
    return os.path.join(set_system_data_directory(system_name), "novella_corpus_representation", "raw_dtm")


def vocab_lists_dicts_directory(system_name):
    """
    :param system_name: "my_mac", "my_xps", or "my_WindowsPC. Here, the respective Computer system with its respective directory structure has to be selected"
    :return: the path for the directory to store all vocab lists, dictionaries such as translation tables, stopword-lists etc
    """
    if system_name == "wcph113":
        return os.path.join(set_system_data_directory(system_name), "vocab_lists_dicts")
    else:
        return os.path.join(set_system_data_directory(system_name), "vocab_lists_dicts")


def mallet_directory(system_name):
    return os.path.join(set_DistReading_directory(system_name), "mallet")

def language_model_path(system_name):
    path = os.path.join(local_temp_directory(system_name), "language_models", "my_model_de")
    return path


def conll_base_directory(system_name):
    """
    :param system_name: "my_mac", "my_xps", or "my_WindowsPC. Here, the respective Computer system with its respective directory structure has to be selected"
    :return: the path for the directory to store all vocab lists, dictionaries such as translation tables, stopword-lists etc
    """
    if system_name == "wcph113":
        return "/mnt/data/users/schroeter/llpro_testing/files/out"
    else:
        return "/mnt/data/users/schroeter/llpro_testing/files/out"


"""
2. Load preprocessing files such as stop word lists etc.
"""

def load_stoplist(filepath):
    """
    generates a list with of comma separated terms to be used as a stop word reduction or elimination list. Lowercase all items
    :param filepath: the input file should be plain text file. Terms should be separated by \n
    :return: a list of lower cased comma separated terms that can be used as elimination or reduction list
    """
    with open(filepath, "r", encoding="utf-8") as infile:
        text = infile.read()
    stopword_list = list(map(str, list(text.split("\n"))))
    stopword_list = [x for x in stopword_list if x]
    return stopword_list

def merge_several_stopfiles_to_list(list_of_filepaths):
    """
    Concatenates the items from several stopwordlist files to one stopword list
    :param list_of_filepaths: a list of filepaths of the stopword files
    :return: an ordered list. Dubletten are eliminated
    """
    global_list = []
    for filepath in list_of_filepaths:
        current_list = load_stoplist(filepath)
        for item in current_list:
            global_list.append(item)
    global_list = list(set(global_list))
    return sorted(global_list)

def save_stoplist(stopword_list, outfilepath):
    """
    Saves a stopword list (items separated by comma in list) in a stopword list with items separated by \n
    :param stopword_list: list of stopwords to be saved as stopword file
    :param outfilepath: filepath for the stopword file
    :return: stopword file as txt file in outfilepath with items separated by \n,
    """
    text = '\n'.join(map(str, stopword_list))
    with open(outfilepath, "w", encoding="utf-8") as outfile:
        outfile.write(text)


def word_translate_table_to_dict(infile_path, also_lower_case=True):
    """
    returns a dictionary based on a txt file with the following structure of word pairs: word_to_be_translated, translation for each pair, with each pair in a separate line
    """
    with open(infile_path, "r", encoding="utf-8") as infile:
        normalization_table = infile.read()
    normalization_dict = {}
    normalization_lower_dict = {}
    for line in normalization_table.splitlines():
        print(line)
        old, new = line.split(",")
        normalization_dict[old] = new
        old_lower = old.lower()
        new_lower = new.lower()
        normalization_lower_dict[old_lower] = new_lower
    if also_lower_case == True:

        return normalization_dict, normalization_lower_dict
    else:
        return normalization_dict


def keywords_to_semantic_fields(list_of_keywords, n_most_relevant_words, vocabulary_path, spacy_model):
    """
    generates a list of words of a semantic field from a list of keywords, based on word embeddings in a spacy model (most similar word vectors)
    """
    nlp = spacy.load(spacy_model)
    all_output_words = []

    for word in list_of_keywords:
        ms = nlp.vocab.vectors.most_similar(
            np.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]), n=50)
        words = [nlp.vocab.strings[w] for w in ms[0][0]]
        all_output_words.extend(words)

    vocab_list = load_stoplist(vocabulary_path)
    all_output_words_reduced = [word for word in all_output_words if word in vocab_list]
    all_words_string = " ".join(all_output_words_reduced)
    doc = nlp(all_words_string)
    lemma_list = [token.lemma_ for token in doc]
    word_counter = Counter(lemma_list)
    words_list = [word for word, count in word_counter.most_common(n_most_relevant_words)]
    print(words_list)
    return words_list