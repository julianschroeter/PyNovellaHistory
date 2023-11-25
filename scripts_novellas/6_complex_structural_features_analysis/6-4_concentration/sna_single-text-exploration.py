from preprocessing.sna import NEnetwork
from preprocessing.text import Text
from preprocessing.presetting import global_corpus_directory
from preprocessing.presetting import global_corpus_representation_directory, language_model_path, vocab_lists_dicts_directory
from preprocessing.presetting import set_DistReading_directory
from preprocessing.corpus import DocNetworkfeature_Matrix
import os
import networkx as nx
import matplotlib.pyplot as plt

system = "my_xps" # "my_mac" # " "my_mac"
my_model_de = language_model_path(system)
normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")

filepath = os.path.join(global_corpus_directory(system_name=system), "00117-00_N_Dorothea_Liebe und Welt_2_Aglaja1832.txt")

char_netw = NEnetwork(filepath=filepath, language_model= my_model_de, minimal_reference=3, text=None, id = None, chunks=None,
                      pos_triples=None, remove_hyphen=True,
                      correct_ocr=True, eliminate_pagecounts=True, handle_special_characters=True, inverse_translate_umlaute=False,
                      eliminate_pos_items=True, keep_pos_items=False, list_keep_pos_tags=None, list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"], lemmatize=True,
                      sz_to_ss=False, translate_umlaute=False, max_length=5000000,
                      remove_stopwords=False, stopword_list=None, token_length=0, normalize_orthogr=True, normalization_table_path=normalization_table_path)

print(char_netw.list_eliminate_pos_tags)
char_netw()
char_netw.f_chunking(segmentation_type="relative",  fixed_chunk_length=1000, num_chunks=5)

char_netw.generate_characters_graph(reduce_to_one_name=True)
proportion_of_characters_with_degcentr1 = char_netw.proportion_of_characters_with_degree()


#print(char_netw.characters_dict["character_tupels_global"])
print("Liste der Figuren:", char_netw.characters_list)
print("Anzahl der Figuren:", char_netw.characters_counter)
print("Paare REl Frq: ",  char_netw.character_pairs_rel_freq)
print("pairs counter: ", char_netw.character_pairs_counter)


G = char_netw.graph

# lassen uns die Werte für den degree centrality je Figur ausgeben:
print("Degree centrality:", nx.degree_centrality(G))

'''Wir sehen, dass das degree centrality Maß die Gewichte nicht berücksichtigt. Hier haben wir 
noch einiges zu tun. Im ersten Schritt können wir aber für jede Figur einmal die absoluten 
Kantengewichte zählen lassen:'''
print("weighted_centrality:", G.degree(weight="weight"))
weighted_tuples = G.degree(weight="weight")
sorted_weighted_tuples = sorted(weighted_tuples, key=lambda tup: tup[1], reverse=True)
print(sorted_weighted_tuples)
sorted_names = [item[0] for item in sorted_weighted_tuples]
print(sorted_names[2:5])
print("degree der Kanten ohne Gewichtung:", G.degree(weight="None"))
print("density:", nx.density(G))
print("nx.degree", nx.degree(G, weight="weight"))
print("Degree histogram", nx.degree_histogram(G))
print("group degree centrality wichtigste Figuren", nx.group_degree_centrality(G, sorted_names[2:5] ))
print("ID:", char_netw.id)
print("text length in tokens", char_netw.token_length)
# Abschließend erzeugen wir eine Visualisierung des Graphen und speichern diesen ab:
nx.draw(G, with_labels=True, font_weight='bold')

plt.savefig(os.path.join(set_DistReading_directory(system), "data", "network_graph.png"))