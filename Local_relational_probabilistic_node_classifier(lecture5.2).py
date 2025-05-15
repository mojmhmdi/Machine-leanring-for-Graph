
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

# load G
G = nx.karate_club_graph()
# nx.draw(G, with_labels = True)

nodes = G.nodes()
edges = G.edges()

labels = nx.get_node_attributes(G, 'club')
print(labels)


labels = {node: 1 if labels[node] == 'Mr. Hi' else 0 for node in nodes}
labels = {node: (0.5 if (1 < np.random.randint(10, 21) < 15.5 ) else label) for node, label in labels.items()}
target = torch.tensor(list(labels.values()), dtype = torch.float32).reshape(1,-1)

indices_labeled = torch.tensor([node for node, label in labels.items() if label != 0.5])
indices_unlabeleds = torch.tensor([node for node, label in labels.items() if label == 0.5])

masker_labeled = torch.eye(len(labels), dtype=torch.float32)
masker_labeled[:, indices_labeled] = 0

masker_unlabeled = torch.eye(len(labels), dtype=torch.float32)
masker_unlabeled[:, indices_unlabeleds] = 0

adjacency_matrix = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
masked_adjacency = torch.matmul(adjacency_matrix, masker_labeled )

container = []
for i in range(100):
    masked_prob = torch.matmul(target, masked_adjacency) / adjacency_matrix.sum(axis=0, keepdims=True).reshape(1,-1)
    final_prob = masked_prob + (masker_unlabeled @ target.T).reshape(1,-1)
    target = final_prob
    container.append(final_prob.sum())

plt.plot(container, 'o')

final_class_probability = target

