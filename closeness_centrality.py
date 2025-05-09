import networkx as nx
import numpy as np
import torch
# load G
G = nx.karate_club_graph()
nx.draw(G, with_labels = True)

# closeness centrality 
closeness_centrality = nx.closeness_centrality(G)

def closeness_centrality(G, node = 0):
    return (len(G.nodes)-1)/(nx.closeness_centrality(G)[node])

print(closeness_centrality(G, 0))
