import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = nx.karate_club_graph()
def add_node_features(G):
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    betweenness = nx.betweenness_centrality(G)
    for node in G.nodes():
        G.nodes[node]['degree'] = degrees[node]
        G.nodes[node]['clustering'] = clustering[node]
        G.nodes[node]['betweenness'] = betweenness[node]

add_node_features(G)

features = []
labels = []
for node, data in G.nodes(data=True):
    features.append([data['degree'], data['clustering'], data['betweenness']])
    labels.append(0 if data['club'] == 'Mr. Hi' else 1)

features = torch.tensor(features, dtype=torch.float32, device=device)
labels = torch.tensor(labels, dtype=torch.long, device=device)

train_idx, test_idx = train_test_split(range(len(features)), test_size=0.4, random_state=42)
train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

train_features, train_labels = features[train_idx], labels[train_idx]
test_features, test_labels = features[test_idx], labels[test_idx]

class NodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

input_dim, hidden_dim, output_dim = 3, 16, 2
model = NodeClassifier(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs = model(test_features)
    test_preds = test_outputs.argmax(dim=1)
    test_acc = (test_preds == test_labels).float().mean().item()
    print(f'Test Accuracy: {test_acc:.4f}')

new_labels = labels.clone()
new_labels[test_idx] = test_preds

num_nodes, num_classes = len(G.nodes()), 2
def compute_neighborhood_summary(G, labels):
    summary = torch.zeros((num_nodes, num_classes), device=device)
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            summary[node, labels[neighbor]] += 1
    return summary

neighborhood_summary = compute_neighborhood_summary(G, new_labels)

class EnhancedNodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

input_dim = 3 + num_classes
enhanced_model = EnhancedNodeClassifier(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(enhanced_model.parameters(), lr=0.01)

k = 15
for iteration in range(k):
    combined_features = torch.cat((features, neighborhood_summary), dim=1)
    for epoch in range(50):
        enhanced_model.train()
        optimizer.zero_grad()
        outputs = enhanced_model(combined_features[train_idx])
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
    enhanced_model.eval()
    with torch.no_grad():
        final_outputs = enhanced_model(combined_features[test_idx])
        final_preds = final_outputs.argmax(dim=1)
        final_acc = (final_preds == test_labels).float().mean().item()
        print(f"Final Test Accuracy after {iteration + 1} iterations: {final_acc:.4f}")
        all_outputs = enhanced_model(combined_features)
        all_preds = all_outputs.argmax(dim=1)
        new_labels[test_idx] = all_preds[test_idx]
        neighborhood_summary = compute_neighborhood_summary(G, new_labels)
