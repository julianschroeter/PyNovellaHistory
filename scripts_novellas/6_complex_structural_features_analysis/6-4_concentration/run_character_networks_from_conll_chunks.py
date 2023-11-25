import pandas as pd

system = "my_xps"

import os
from preprocessing.presetting import local_temp_directory
from itertools import combinations
from collections import Counter
import networkx as nx
from math import sqrt
import matplotlib.pyplot as plt
import re
from preprocessing.sna import get_centralization


corpus_dir = os.path.join(local_temp_directory(system), "Namen_novellas_chunks")

doc_ids = []

for filename in os.listdir(corpus_dir):
    doc_ids.append(filename[:5])

doc_ids = list(set(doc_ids))
print(doc_ids)

doc_dic = {}
for id in doc_ids:
    files_list = []
    for filename in os.listdir(corpus_dir):

        if id in filename:
            files_list.append(filename)
    doc_dic[id] = sorted(files_list)


comb_dic = {}
names_dic = {}
for doc_id, chunks in doc_dic.items():
    doc_id = str(doc_id)+"-00"
    combs_list = []
    all_names = []
    for filename in os.listdir(corpus_dir):
        if filename in chunks:
            names = open(os.path.join(corpus_dir, filename), "r", encoding="utf8").read()
            names = names.split(" ")

            corr_names = []
            for name in names:
                if re.search("'s$", name):
                    red_name = name[:-2]
                    print(red_name)
                    print(name)
                    if red_name in names:
                        print(name)
                        print(red_name)
                        name = red_name


                elif re.search("s$", name):
                    red_name = name[:-1]
                    print(name)
                    print(red_name)
                    if red_name in names:
                        print(name)
                        print(red_name)
                        name = red_name
                elif re.search("s[,.;]?$", name):
                    red_name = name[:-2]
                    print(name)
                    print(red_name)
                    if red_name in names:
                        print(name)
                        print(red_name)
                        name = red_name

                elif re.search("s'[,.;]?$", name):
                    red_name = name[:-3]
                    print(name)
                    print(red_name)
                    if red_name in names:
                        print(name)
                        print(red_name)
                        name = red_name

                corr_names.append(name)

            names = list(set(corr_names))
            pairs_in_chunk = list(combinations(names, 2))
            pairs_in_chunk = [tuple(sorted(pair)) for pair in pairs_in_chunk]
            combs_list.extend(pairs_in_chunk)
            all_names.extend(names)
    all_names = list(set(all_names))
    names_dic[doc_id] = [" ".join(all_names), len(all_names)]
    comb_dic[doc_id] = combs_list

print(comb_dic)
print(names_dic)

centr_dic = {}

plots_directory = os.path.join(local_temp_directory(system), "figures", "conll_network_plots2")

if not os.path.exists(plots_directory):
    os.makedirs(plots_directory)

for id, combs in comb_dic.items():

    character_pairs_counter = Counter(combs)
    character_pairs_rel_freq = [(tuple[0], tuple[1], (count/5)) for tuple, count in character_pairs_counter.items() if count >= 2]
    G = nx.Graph()
    G.add_weighted_edges_from(character_pairs_rel_freq)
    density = nx.density(G)
    degree_centrality = nx.degree_centrality(G)
    if len(degree_centrality) == 0:
        degree_centrality = {"NO_CHARACTER":0}

    weighted_degree_centrality = nx.degree(G, weight="weight")
    centralization = get_centralization(degree_centrality, "degree")
    max_deg_centr = max(degree_centrality.values())
    centr_dic[id] = [degree_centrality, weighted_degree_centrality, density, centralization, max_deg_centr]

    nx.draw(G,with_labels=True, font_weight='bold')
    filename = str(id + ".png")
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

df.to_csv(os.path.join(local_temp_directory(system), "conll_based_networkdata-matrix-novellas.csv"))
#df = df.rename(columns={0:"degree_centrality",1: "weighted_degree_centrality",2: "density",
 #                       3:"Figuren",4: "Figurenanzahl"})


