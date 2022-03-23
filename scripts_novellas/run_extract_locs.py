system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')


from preprocessing.presetting import global_corpus_directory, global_corpus_representation_directory, language_model_path, vocab_lists_dicts_directory
from preprocessing.corpus import DocNetworkfeature_Matrix, DocNEs_Matrix

import os


my_model_de = language_model_path(system) # "de_core_news_lg"

corpus_path = global_corpus_directory(system, test=False)
normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")

outfile_path_df = os.path.join(global_corpus_representation_directory(system), "NEs_document_Matrix_test.csv")
outfile_path_characters_list = os.path.join(global_corpus_representation_directory(system), "corpus_characters_list.txt")
outfile_path_locs_list = os.path.join(global_corpus_representation_directory(system), "corpus_locs_list.txt")

ne_matrix_object = DocNEs_Matrix(corpus_path=corpus_path, language_model=my_model_de, segmentation_type="fixed", normalize_orthogr=False,
                                 normalization_table_path=None,keep_pos_items=False)

ne_matrix_object.generate_df()
ne_matrix_object.save_csv(outfile_path= outfile_path_df)
print(ne_matrix_object.corpus_locs_list)
