import os
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
    :param system_name: "my_mac", "my_xps", "wcph113" or "wcph104". Here, the respective Computer system with its respective directory structure has to be selected"
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

def global_corpus_directory(system_name, test=False):
    """
    :param system_name:
    "my_mac", "my_xps", or "my_WindowsPC. Here, the respective Computer system with its respective directory structure has to be selected"
    "test": If True, the directory for a samm test sample is selected; if False: the whole corpus is selected.
    :return: the path for the directory with all plain text files of the project corpus for the system specified with the parameter
    """
    if test == False:
            return os.path.join(set_system_data_directory(system_name), "novella_corpus_all")
    elif test == True:
            return os.path.join(set_system_data_directory(system_name), "novella_corpus_test")

def global_corpus_representation_directory(system_name):
    """
        :param system_name: "my_mac", "my_xps", or "my_WindowsPC. Here, the respective Computer system with its respective directory structure has to be selected"
        :return: the path for the directory to store all corpus representation files such as dtms or lists for the system specified with the parameter
        """
    return os.path.join(set_DistReading_directory(system_name),"data", "novella_corpus_representation")


def global_corpus_raw_dtm_directory(system_name):
    """
        :param system_name: "my_mac", "my_xps", or "my_WindowsPC. Here, the respective Computer system with its respective directory structure has to be selected"
        :return: the path for the directory to store all corpus representation files such as dtms or lists for the system specified with the parameter
        """
    return os.path.join(set_system_data_directory(system_name), "novella_corpus_representation", "raw_dtm")


def mallet_directory(system_name):
    return os.path.join(set_DistReading_directory(system_name), "mallet")

def language_model_path(system_name, lang="de"):
    if system_name == "my_mac" and lang == "de":
        path = os.path.join(local_temp_directory(system_name), "language_models", "my_model_de")
    else:
        pass
    return path



"""
2. Load preprocessing files such as stop word lists etc.
"""

def load_stoplist(filepath):
    """
    generates a list with of comma separated terms to be used as a stop word reduction or elimination list. Lowercase all items
    :param filepath: the input file should be plain text file. Terms should be separated by \n
    :return: a list of lower cased comma separated terms that can be used as elimination or reduction list
    """
    with open(filepath, "r", encoding="utf8") as infile:
        text = infile.read()
    stopword_list = list(map(str.lower, list(text.split("\n"))))
    stopword_list = [x for x in stopword_list if x]
    return stopword_list

def merge_several_stopfiles_to_list(list_of_filepaths):
    """
    Concatenates the items from several stopwordlist files to one stopword list
    :param list_of_filepaths: a list of filepaths of the stopword files
    :return: an ordered list
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
    with open(outfilepath, "w", encoding="utf8") as outfile:
        outfile.write(text)