import torch
import torch_geometric

import networkx as nx
import matplotlib.pyplot as plt
from obth_gnn.data import MaterialGraph

def graph_info(data):
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')  # Number of nodes in the graph
    print(f'Number of edges: {data.num_edges}')  # Number of edges in the graph
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')  # Average number of edges/node in the graph
    print(
        f'Contains isolated nodes: {data.has_isolated_nodes()}')  # Does the graph contains nodes that are not connected
    print(
        f'Contains self-loops: {data.has_self_loops()}')  # Does the graph contains nodes that are linked to themselves
    print(f'Is undirected: {data.is_undirected()}')  # Is the graph an undirected graph
def create_custom_graph():
    # Node features
    x = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.float)

    # Edge indices
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)

    # Edge features
    edge_attr = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]], dtype=torch.float)

    # Global features
    u = torch.tensor([0.9, 1.0], dtype=torch.float)

    # Create custom graph
    graph = MaterialGraph(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)
    return graph

# Usage
starting_graph = create_custom_graph()
print(starting_graph)

for key, item in starting_graph:
    print(f'{key} found in data')






g = torch_geometric.utils.to_networkx(custom_graph, to_undirected=False)
nx.draw(g)
plt.show()
