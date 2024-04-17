import pandas as pd

system = "my_xps"

import os
from preprocessing.presetting import local_temp_directory, global_corpus_representation_directory
from itertools import combinations
from collections import Counter
import networkx as nx
from math import sqrt
import matplotlib.pyplot as plt
import re
from preprocessing.sna import get_centralization


corpus_dir = os.path.join(local_temp_directory(system), "conll_Namen_novellas")
plots_directory = os.path.join(local_temp_directory(system), "figures", "conll_network_plots_15mostcommon")
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system), "Bibliographie.csv")

meta_df = pd.read_csv(metadata_csv_filepath, index_col=0)
author_names = list(set(meta_df["Nachname"].values.tolist() + meta_df["Vorname"].values.tolist()+meta_df["Pseudonym"].values.tolist()))
print(author_names)
comb_dic = {}
names_dic = {}

for filename in os.listdir(corpus_dir):
    print(filename)
    names = open(os.path.join(corpus_dir, filename), "r", encoding="utf8").read()
    names = names.split(" ")
    corr_names = []
    word_count = 0
    for name in names:
        if re.search("'s$", name):
            red_name = name[:-2]
            print(red_name)
            print(name)
            if red_name in names:
                print(name)
                print(red_name)
                name = red_name
        if word_count < 4:
            if name not in author_names:
                corr_names.append(name)
        else:
            corr_names.append(name)
        word_count += 1

    names = corr_names

    n_most_common = 15
    names_most_common = Counter(names).most_common(n_most_common)
    names_most_common = [name for name, count in names_most_common]

    names = [name for name in names if name in names_most_common]

    num_chunks = 5
    chunk_length = int(len(names) / num_chunks) + (len(names) % num_chunks > 0)  # round to the next integer

    chunks = []
    current_chunk = []
    current_chunk_count = 0
    for name in names:
        current_chunk_count += 1
        current_chunk.append(name)
        if current_chunk_count == chunk_length:
            current_chunk_str = " ".join(current_chunk)
            chunks.append(current_chunk_str)
            # start for the next chunk
            current_chunk = []
            current_chunk_count = 0
    final_chunk_str = " ".join(current_chunk)
    if final_chunk_str.strip() and final_chunk_str:
        chunks.append(final_chunk_str)
    combs_list = []
    all_names = []
    for chunk in chunks:
        chunk_names = list(set(chunk.split(" ")))
        pairs_in_chunk = list(combinations(chunk_names, 2))
        pairs_in_chunk = [tuple(sorted(pair)) for pair in pairs_in_chunk]
        combs_list.extend(pairs_in_chunk)
        all_names.extend(chunk_names)
    all_names = list(set(all_names))


    names_dic[filename] = [" ".join(all_names), len(all_names)]
    comb_dic[filename] = combs_list

print(comb_dic)
print(names_dic)

centr_dic = {}


if not os.path.exists(plots_directory):
    os.makedirs(plots_directory)

for id, combs in comb_dic.items():

    character_pairs_counter = Counter(combs)
    print(id)
    print(combs)
    print(character_pairs_counter)
    character_pairs_rel_freq = [(tuple[0], tuple[1], (count/5)) for tuple, count in character_pairs_counter.items() if count >= 1]
    G = nx.Graph()
    G.add_weighted_edges_from(character_pairs_rel_freq)
    density = nx.density(G)
    degree_centrality = nx.degree_centrality(G)
    if len(degree_centrality) == 0:
        degree_centrality = {"NO_CHARACTER":0}
    print(id)
    print(degree_centrality)

    weighted_degree_centrality = nx.degree(G, weight="weight")
    centralization = get_centralization(degree_centrality, "degree")
    max_deg_centr = max(degree_centrality.values())
    centr_dic[id] = [degree_centrality, weighted_degree_centrality, density, centralization, max_deg_centr]

    nx.draw(G,with_labels=True, font_weight='bold')
    filename = str(id + ".svg")
    print(filename)
    plt.title(filename)
    plt.savefig(os.path.join(plots_directory, filename))
    plt.show()
    plt.clf()

print(centr_dic)

import pandas as pd

df = pd.DataFrame.from_dict(centr_dic, orient="index", columns=["degree_centrality", "weighted_deg_centr", "density", "centralization",
                                                                "max_degree_centrality"])

names_df = pd.DataFrame.from_dict(names_dic, orient="index", columns = ["Figuren", "Figurenanzahl"])

df = pd.concat([df, names_df], axis=1)
print(df)

df.to_csv(os.path.join(local_temp_directory(system), "conll_based_networkdata-matrix-novellas_15mostcommon.csv"))
#df = df.rename(columns={0:"degree_centrality",1: "weighted_degree_centrality",2: "density",
 #                       3:"Figuren",4: "Figurenanzahl"})


