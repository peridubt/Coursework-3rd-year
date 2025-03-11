import networkx as nx
import torch
import osmnx as ox
from torch_geometric.data import Data

# Create a sample MultiDiGraph
G = ox.graph_from_bbox((39.121447, 51.646002, 39.135578, 51.653782))

# Extract nodes and their attributes
nodes = list(G.nodes(data=True))
node_features = [node[1]['attr'] for node in nodes]

# Extract edges and their attributes
edges = list(G.edges(data=True))
edge_index = []
edge_attr = []
for edge in edges:
    edge_index.append([edge[0], edge[1]])
    edge_attr.append(edge[2])

# Convert lists to tensors
node_features = torch.tensor([ord(c) for c in ''.join(node_features)])  # Example conversion
edge_index = torch.tensor(edge_index).T
edge_attr = torch.tensor([ord(c) for c in ''.join(edge_attr)])  # Example conversion
