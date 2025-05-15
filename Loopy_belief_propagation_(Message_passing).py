import numpy as np

def normalize(vec):
    vec = vec.astype(np.float64)
    s = np.sum(vec, dtype=np.float64)
    return vec / s if s != 0 else np.ones_like(vec, dtype=np.float64) / len(vec)

nodes = [chr(ord('A') + i) for i in range(20)]

adjacency = {}
for i, node in enumerate(nodes):
    neighbors = []
    if i > 0:
        neighbors.append(nodes[i - 1])
    if i < len(nodes) - 1:
        neighbors.append(nodes[i + 1])
    adjacency[node] = neighbors

phi = {node: np.array([0.5, 0.5], dtype=np.float64) for node in nodes}
phi[nodes[0]] = np.array([0.0, 1.0], dtype=np.float64)

psi = np.array([
    [1.0, 0.5],
    [0.5, 1.0]
], dtype=np.float64)
iterations = 18

k = psi.shape[0]
messages = {
    i: {j: np.ones(k, dtype=np.float64) for j in adjacency[i]}
    for i in nodes
}
 
beliefs = {i: normalize(phi[i].copy()) for i in nodes}

for _ in range(iterations):
    new_messages = {i: {} for i in nodes}
    for i in nodes:
        for j in adjacency[i]:
            prod = np.ones(k, dtype=np.float64)
            for n in adjacency[i]:
                if n != j:
                    prod *= messages[n][i]
            m_ij = np.zeros(k, dtype=np.float64)
            for xj in range(k):
                for xi in range(k):
                    m_ij[xj] += phi[i][xi] * psi[xi, xj] * prod[xi]
            new_messages[i][j] = normalize(m_ij)
    messages = new_messages

    for i in nodes:
        b = phi[i].copy()
        for n in adjacency[i]:
            b *= messages[n][i]
        beliefs[i] = normalize(b)

for node, belief in beliefs.items():
    print(f"Node {node}: P(0)={belief[0]:.17g}, P(1)={belief[1]:.17g}")
