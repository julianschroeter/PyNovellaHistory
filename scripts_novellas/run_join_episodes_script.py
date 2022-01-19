system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os

from preprocessing.presetting import global_corpus_directory, language_model_path, local_temp_directory
from preprocessing.prepare_corpus import join_episodes

corpus_path = os.path.join(local_temp_directory(system), "pantheon_roh")
my_model_de_path = language_model_path(system)
output_directory = os.path.join(local_temp_directory(system), "pantheon_joined")

join_episodes(corpus_path, output_directory, language_model=language_model_path(system), )