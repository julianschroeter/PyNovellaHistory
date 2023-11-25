import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

B = nx.Graph()

# Add nodes with the node attribute "bipartite"

B.add_nodes_from([1, 2, 3, 4], bipartite=0)

B.add_nodes_from(["a", "b", "c"], bipartite=1)

# Add edges only between nodes of opposite node sets

B.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])

nx.draw(B)
plt.show()

G = nx.Graph()
G.add_node("Figur A", pos = (0.3,0))
G.add_node("Figur B", pos = (0.8,0))
G.add_node("Figur C", pos = (1,1))
G.add_node("Figur D", pos = (0,1))
G.add_node("Ereignis 1", pos= (0.5, 0.5))
G.add_edges_from([("Figur A", "Ereignis 1"),
                  ("Figur A", "Figur B"),
                  ("Figur A", "Figur C"),
                  ("Figur A", "Figur D"),
                  ("Figur B", "Ereignis 1"),
                  ("Figur B", "Figur C"),
                  ("Figur B", "Figur D"),
                  ("Figur C", "Ereignis 1"),
                ("Figur C", "Figur D"),
                  ("Figur D", "Ereignis 1")
                              ])
pos = nx.get_node_attributes(G, "pos")

plt.title("Modell eines Ereignis-Figuren-Netzwerks")
nx.draw(G, pos, node_color=["blue", "blue", "blue", "blue","red"],
        edge_color= ["red", "blue", "blue", "blue",
                     "red", "blue","blue","red", "blue", "red"
                      ],
        with_labels=True)

plt.show()