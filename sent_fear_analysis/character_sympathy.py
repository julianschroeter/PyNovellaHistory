system = "wcph113"

system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')


import os
import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, heftroman_base_directory, load_stoplist, vocab_lists_dicts_directory, local_temp_directory
from ast import literal_eval

conll_path = os.path.join(heftroman_base_directory(system), "conll")

def sentence_counter(input_list):
    sentence_count = 0
    sentence_count_list = []
    for i in input_list:
        if i == 0:
            sentence_count += 1
            sentence_count_list.append(sentence_count)
        else:
            sentence_count_list.append(sentence_count)
    return sentence_count_list

def calculate_character_symp(text_df, list_of_characters, pos_list, neg_list):
    """

    :param text_df:
    :param list_of_characters:
    :param pos_list:
    :param neg_list:
    :return: a dictionary with names of character as keys and a degree of likeability as value (>1 more likeable, < more unlikeable)
    """

    input_values = text_df["nr"].values.tolist()
    sentence_counts = sentence_counter(input_values)
    text_df["sentence_nr"] = sentence_counts

    grouped = text_df.groupby("sentence_nr")
    dfs_list = [grouped.get_group(df) for df in grouped.groups]

    symp_dic = {}

    for name in list_of_characters:
        print(name)
        if name == "JohnSinclair":
            dummy_name = "John"
        elif name == "JerryCotton":
            dummy_name = "Jerry"
        elif name == "PerryRhodan":
            dummy_name = "Perry"
        elif name == "TomShark":
            dummy_name = "Tom"
        elif name == "ReginaldBull":
            dummy_name = "Bully"
        else:
            dummy_name = name
        pos_rate, neg_rate = 0, 0
        for df in dfs_list:

            sent_lemmas = df["Lemma"].values.tolist()

            if dummy_name in sent_lemmas:
                print("Figurenname: ",dummy_name, "; ",name)
                for word in sent_lemmas:
                    if word in pos_list:
                        print("positive word: ", word)
                        print("in sentence: ", sent_lemmas)
                        pos_rate += 1
                    elif word in neg_list:
                        print("negative word: ", word)
                        print("in sentence: ", sent_lemmas)
                        neg_rate += 1

        print(pos_rate, neg_rate)
        if neg_rate + pos_rate != 0:
            symp_rate =  pos_rate / (pos_rate + neg_rate)
        else:
            symp_rate = 0

        symp_dic[name] = symp_rate
        print(symp_dic)

    return symp_dic


def add_symp_df(df, pos_list, neg_list):
    df["id"] = df.index.values
    print(df)
    df["symp_dict"] = df.apply(lambda x : calculate_character_symp(pd.read_csv(os.path.join(conll_path, str(x.id)),engine="python" , sep="\t", on_bad_lines="skip",names= ["nr", "Token", "Lemma", "?_", "POS", "Casus", "id", "pars", "_", "Coref"]), str(x.Figuren).split(". "), pos_list, neg_list), axis=1)
    return df


sna_matrix_path = os.path.join(global_corpus_representation_directory(system), "Heftromane_SNA_Matrix.csv")

sna_df = pd.read_csv(sna_matrix_path, index_col=0)

print(sna_df)

pos_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "ResultList_pos-Charakter.txt"))
neg_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "ResultList_neg-Charakter.txt"))

emo_df = pd.read_csv(os.path.join(vocab_lists_dicts_directory(system), "emotion_lexicon.csv"))
pos_df = emo_df[emo_df["Positive"] == 1]
neg_df = emo_df[emo_df["Negative"] == 1]
pos_list = pos_df["German (de)"].values.tolist()
neg_list = neg_df["German (de)"].values.tolist()


full_sna_df = add_symp_df(sna_df, pos_list, neg_list)
print(full_sna_df)

full_sna_df.to_csv(os.path.join(local_temp_directory(system), "Heftromane_SNA_Emo_Matrix.csv"))
