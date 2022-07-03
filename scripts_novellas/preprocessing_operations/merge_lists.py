system = "wcph113"

from preprocessing.presetting import merge_several_stopfiles_to_list, vocab_lists_dicts_directory, global_corpus_representation_directory, save_stoplist
import os
filepath = vocab_lists_dicts_directory(system)
file1 = os.path.join(filepath, "ergaenzungen_german_list.txt")
file2 = os.path.join(global_corpus_representation_directory(system), "wordlist_german.txt")
global_list = merge_several_stopfiles_to_list([file2, file1], )

if "prinzessin" in global_list:
    print("prinzessin in global list")
print(global_list[:1000])

save_stoplist(global_list, os.path.join(global_corpus_representation_directory(system), "ext_wordlist_german.txt"))
print("saved")
