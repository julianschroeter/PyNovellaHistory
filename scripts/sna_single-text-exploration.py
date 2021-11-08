from preprocessing.SNA import CharacterNetwork
from preprocessing.text import Text
from preprocessing.presetting import global_corpus_directory
from preprocessing.presetting import global_corpus_representation_directory
from preprocessing.presetting import set_DistReading_directory
from preprocessing.corpus import DocNetworkfeature_Matrix
import os
import networkx as nx
import matplotlib.pyplot as plt

system = "wcph104" # "my_xps" "my_mac"
my_model_de = os.path.join(set_DistReading_directory(system_name=system), "language_models", "model_de_lg")
filepath = os.path.join(global_corpus_directory(system_name=system), "00022-00_N_Tieck_Ludwig_Des_Lebens_Ueberfluss_1838UraniaTB_DTNovSchatz_3KANON.txt")

char_netw = CharacterNetwork(filepath=filepath, minimal_reference=2, text=None, id = None, chunks=None, pos_triples=None, remove_hyphen=True,
                 correct_ocr=True, eliminate_pagecounts=True, handle_special_characters=True, inverse_translate_umlaute=False,
                 eliminate_pos_items=True, keep_pos_items=False, list_keep_pos_tags=None, list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"], lemmatize=False,
                 sz_to_ss=False, translate_umlaute=False, max_length=5000000,
                 remove_stopwords="before_chunking", stopword_list=None, language_model= my_model_de)


char_netw()
char_netw.f_chunking(segmentation_type="paragraph",  fixed_chunk_length=1000, num_chunks=5)
#print(char_netw.chunks)
char_netw.generate_characters_dict()
#print(char_netw.characters_dict["character_tupels_global"])
print("Liste der Figuren:", char_netw.characters_dict["characters"])
print("Anzahl der Figuren:", char_netw.characters_dict["character_counter"])
print(char_netw.characters_dict["character_tuples_counter"])
print(char_netw.characters_dict["character_tuples_relative"])

G = nx.Graph()
# fügen ihm die gewichteten Kanten zwischen Knoten (Figuren) hinzu:
G.add_weighted_edges_from(char_netw.characters_dict["character_tuples_relative"])

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
print("text_length in tokens", char_netw.token_length)
# Abschließend erzeugen wir eine Visualisierung des Graphen und speichern diesen ab:
nx.draw(G, with_labels=True, font_weight='bold')
plt.savefig(os.path.join(set_DistReading_directory(system), "data", "degree_centrality_graph.png"))