
import numpy as np

np.random.seed(42)
adjacency_matrix = np.random.randint(0, 2, size=(34, 34))
np.fill_diagonal(adjacency_matrix, 0)
col_sums = adjacency_matrix.sum(axis=0, keepdims=True)
col_sums[col_sums == 0] = 1
normalized_adjacency = adjacency_matrix / col_sums

beta = 0.8
google_matrix = beta * normalized_adjacency + (1 - beta) * (np.ones((34, 34)) / 34)

pagerank = np.ones(34) / 34  
pagerank = google_matrix @ pagerank  

iter = 10

for i in range (iter):
    pagerank = google_matrix @ pagerank  

print(f"PageRank vector after {iter} iterations:", pagerank)

np.array_equal(pagerank, np.ones(34) / 34)
