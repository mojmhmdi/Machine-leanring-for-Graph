
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx
from torch import sigmoid
from torch.optim import Adam
import numpy as np


print(torch.__version__)


G = nx.karate_club_graph()

def visualize_emb(emb):
  X = emb.weight.data.numpy()
  pca = PCA(n_components=2)
  components = pca.fit_transform(X)
  plt.figure(figsize=(6, 6))
  club1_x = []
  club1_y = []
  club2_x = []
  club2_y = []
  for node in G.nodes(data=True):
    if node[1]['club'] == 'Mr. Hi':
      club1_x.append(components[node[0]][0])
      club1_y.append(components[node[0]][1])
    else:
      club2_x.append(components[node[0]][0])
      club2_y.append(components[node[0]][1])
  plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
  plt.scatter(club2_x, club2_y, color="blue", label="Officer")
  plt.legend()
  plt.show()

weighted_adj = torch.tensor(nx.to_numpy_array(G), dtype=torch.long)
adj_torch = torch.tensor(weighted_adj/(weighted_adj + torch.finfo(torch.float32).eps))
 
 
def create_node_embedding(n_nodes, emb_dim):
    embedding = nn.Embedding(num_embeddings=n_nodes, embedding_dim=emb_dim)
    embedding.weight.data = torch.rand([n_nodes, emb_dim])
    embedding.weight.requires_grad = True
    return embedding

def  loss_function(logits, labels):
    epsilon = 1e-12  
    prob = torch.sigmoid(logits)
    prob = torch.clamp(prob, min = epsilon, max = 0.9999)
    loss = -(labels * torch.log (prob) + (1 - labels) * torch.log(1 - prob))
    if torch.mean(loss).item() > 1e108:
        plt.plot(prob.flatten().cpu().detach().numpy())
    mask = ~torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    loss = loss[mask]
    
    return torch.mean(loss)

weighted_adj = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
adj_matrix = torch.tensor(weighted_adj/(weighted_adj + torch.finfo(torch.float32).eps), dtype = torch.long)
adj_matrix.requires_grad_(False)

embeddings = create_node_embedding(34, 16)
optimizer = Adam(embeddings.parameters(), lr=0.01)

num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    # logits = torch.matmul(embeddings.weight, embeddings.weight.T)
    loss = loss_function(torch.matmul(embeddings.weight, embeddings.weight.T), adj_matrix)
    loss.backward() 
    optimizer.step()
    # print(epoch)
    if epoch % 10 == 0 and epoch != 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

visualize_emb(embeddings)

