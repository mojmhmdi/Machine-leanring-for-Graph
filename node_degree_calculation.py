
import networkx as nx
import numpy as np
import torch
# load G
G = nx.karate_club_graph()
nx.draw(G, with_labels = True)


#  node degree
def compute_node_degrees(G):
    nodes_list = list(G.nodes())
    edge_list = list(G.edges())
    nodes_degree = np.zeros(len(nodes_list))

    number_of_nodes = len(nodes_list)
    number_of_edges = len(edge_list)

    for i in range(number_of_edges):
        nodes_degree[edge_list[i][0]] = nodes_degree[edge_list[i][0]] +  1
        nodes_degree[edge_list[i][1]] = nodes_degree[edge_list[i][1]] +  1
    return nodes_degree

node_degree = compute_node_degrees(G)
print(sum(compute_node_degrees(G)) / len(G.nodes()))
