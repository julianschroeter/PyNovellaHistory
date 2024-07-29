from preprocessing.sna import get_centralization, scale_centrality


centr_vals = {0:1, 1: 0.2, 2:0.2, 3:0.2,4: 0.2, 5:0.2}
type = "degree"

print(get_centralization(centrality=centr_vals,c_type=type))

