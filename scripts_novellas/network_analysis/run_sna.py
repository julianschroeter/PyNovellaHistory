system = "wcph113" # "my_xps" # "my_mac" # wcph104"

if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')


from preprocessing.presetting import global_corpus_directory, global_corpus_representation_directory, language_model_path, vocab_lists_dicts_directory
from preprocessing.corpus_alt import DocNetworkfeature_Matrix

import os



my_model_de = language_model_path(system)

corpus_path = global_corpus_directory(system, test=False)
outfile_path_df = os.path.join(global_corpus_representation_directory(system), "Networkdata_document_Matrix_test.csv")
outfile_path_characters_list = os.path.join(global_corpus_representation_directory(system), "corpus_characters_list.txt")

normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")

network_matrix_object = DocNetworkfeature_Matrix(corpus_path=corpus_path, language_model=my_model_de, segmentation_type="fixed",
                                                 lemmatize=True, remove_hyphen=True, correct_ocr=True, normalize_orthogr=True,
                                                 normalization_table_path=normalization_table_path,
                                                 reduce_to_words_from_list=False, reduction_word_list=None,
                                                 )
network_matrix_object.generate_df()

network_matrix_object.save_csv(outfile_path= outfile_path_df)
print(network_matrix_object.corpus_characters_list)
network_matrix_object.corpus_characters_list_to_file(outfilepath=outfile_path_characters_list)