system = "my_xps" # "wcph113" #
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import os

from preprocessing.presetting import global_corpus_directory, language_model_path, local_temp_directory, set_system_data_directory
from preprocessing.prepare_corpus import join_episodes

corpus_path = os.path.join(local_temp_directory(system), "episodes_temp" )
my_model_de_path = language_model_path(system)
my_model_de_path = "de_core_news_lg"

output_directory = os.path.join(local_temp_directory(system), "episodes_joined")

join_episodes(corpus_path, output_directory, language_model=my_model_de_path)