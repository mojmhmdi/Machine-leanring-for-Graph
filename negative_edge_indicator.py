
import networkx as nx
import numpy as np
import torch
# load G
G = nx.karate_club_graph()
nx.draw(G, with_labels = True)


# negative edge list
def neg_edges_list_func(G):
        adj_torch = torch.tensor(torch.tensor(nx.to_numpy_array(G), dtype=torch.long)/
                (torch.tensor(nx.to_numpy_array(G), dtype=torch.long) + torch.finfo(torch.float32).eps), dtype = torch.long)
        return  torch.stack(torch.where(adj_torch==0)).T
print(neg_edges_list_func(G))

# a sample of size n from negatice edge list
def get_neg_edges ( G, num_neg_samples = 5):
    
    neg_edge_list = neg_edges_list_func(G)
    return neg_edge_list[torch.randint(0, neg_edge_list.shape[0], (num_neg_samples,))]
print(get_neg_edges(G, 15))

# check if an edge is negative
def can_be_negative(G, edge = [0,0]):
    neg_edge_list = neg_edges_list_func(G)
    return ((neg_edge_list == torch.tensor(edge)).all(dim=1)).any().item()
print(can_be_negative(G, [0,1]))
