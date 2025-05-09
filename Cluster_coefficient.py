
import networkx as nx
import numpy as np
import torch
# load G
G = nx.karate_club_graph()
nx.draw(G, with_labels = True)


# cluster coefficient
def clust_coefficient(G):
    adj_matrix = nx.to_numpy_array(G)/(nx.to_numpy_array(G)+(10e-20))
    adj_square = np.dot(adj_matrix, adj_matrix)
    adj_cube = np.dot(adj_square, adj_matrix)
    return (np.diag(adj_cube) / 2) / ((np.sum(adj_matrix, axis=1) * (np.sum(adj_matrix, axis=1) -1) / 2)
                                      +np.finfo(np.float64).eps)
print(clust_coefficient(G))
