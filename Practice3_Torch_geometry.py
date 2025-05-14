
import os
from torch_geometric.datasets import TUDataset

if 'IS_GRADESCOPE_ENV' not in os.environ:
  root = './enzymes'
  name = 'ENZYMES'

  pyg_dataset= TUDataset(root, name)

  print(pyg_dataset)
  
# Question 1: What is the number of classes and number of features in the ENZYMES dataset? (5 points)

def get_num_classes(pyg_dataset):
  
    return pyg_dataset.num_classes

def get_num_features(pyg_dataset):

    return pyg_dataset.num_features

if 'IS_GRADESCOPE_ENV' not in os.environ:
    num_classes = get_num_classes(pyg_dataset)
    num_features = get_num_features(pyg_dataset)
    print("{} dataset has {} classes".format(name, num_classes))
    print("{} dataset has {} features".format(name, num_features))

# Question 2: What is the label of the graph with index 100 in the ENZYMES dataset? (5 points)

def get_graph_class(pyg_dataset, index):
    return pyg_dataset[index].y.item()

if 'IS_GRADESCOPE_ENV' not in os.environ:
  graph_0 = pyg_dataset[0]
  print(graph_0)
  idx = 6
  label = get_graph_class(pyg_dataset, idx)
  print('Graph with index {} has label {}'.format(idx, label))
  
# Question 3: How many edges does the graph with index 200 have? (5 points)

def get_graph_num_edges(pyg_dataset, index):
    return pyg_dataset[index].edge_index.shape[1]


if 'IS_GRADESCOPE_ENV' not in os.environ:
  idx = 200
  num_edges = get_graph_num_edges(pyg_dataset, idx)
  print('Graph with index {} has {} edges'.format(idx, num_edges))
  
