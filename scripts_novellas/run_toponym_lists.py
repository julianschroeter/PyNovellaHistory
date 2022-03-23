system = "wcph113"

import os

from preprocessing.presetting import load_stoplist, vocab_lists_dicts_directory

italien_toponyms_file = os.path.join(vocab_lists_dicts_directory(system), "toponym_lists", "italian.txt")
french_toponyms_file = os.path.join(vocab_lists_dicts_directory(system), "toponym_lists", "french.txt")
spanish_toponyms_file = os.path.join(vocab_lists_dicts_directory(system), "toponym_lists", "spanish.txt")
german_toponyms_file = os.path.join(vocab_lists_dicts_directory(system), "toponym_lists", "german.txt")


italian_toponyms = load_stoplist(italien_toponyms_file)