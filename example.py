import torch
import torch_geometric

import networkx as nx
import matplotlib.pyplot as plt
from obth_gnn.data import MaterialGraph


def graph_info(data):
    """
    Print graph description.
    :param data: (torch_geometric.data)
    :return: -
    """
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')  # Number of nodes in the graph
    print(f'Number of edges: {data.num_edges}')  # Number of edges in the graph
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')  # Average number of edges/node in the graph
    print(
        f'Contains isolated nodes: {data.has_isolated_nodes()}')  # Does the graph contains nodes that are not connected
    print(
        f'Contains self-loops: {data.has_self_loops()}')  # Does the graph contains nodes that are linked to themselves
    print(f'Is undirected: {data.is_undirected()}')  # Is the graph an undirected graph
    print("Number of features per node:", data.num_features)  # Number of features per node
    print("Number of edge attributes:", data.num_edge_features)  # Number of edge attributes
    print("Number of global attributes:", data.u.shape)  # Number of global attributes


def create_circle_graph(nr_nodes=3, features_node=2, ):
    """
    Constricts a random circular graph
    :param nr_nodes: (int) nr of nodes
    :param features_node: () nr of features/node
    :return:
    """
    # Node features
    x = torch.tensor([[0, 1, 50, 51] for _ in range(nr_nodes)], dtype=torch.float)

    # Edge indices
    ei = [i for i in range(nr_nodes)]  # 0,1,2,3,4
    ef = [i + 1 for i in range(nr_nodes)]  # 1,2,3,0
    ef[-1] = 0
    edges = [[*ei, *ef], [*ef, *ei]]
    print("edges:", edges)
    edge_index = torch.tensor(edges, dtype=torch.long)

    # Edge features
    edge_attr = torch.tensor([[0.1, 0.2, 0.3] for _ in range(2 * len(ei))], dtype=torch.float)

    # Global features
    u = torch.tensor([0.9, 1.0], dtype=torch.float)

    # Create custom graph
    graph = MaterialGraph(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)
    return graph


# Usage
starting_graph = create_circle_graph(5)

print("starting_graph",starting_graph.x)
print("data.edge_index",starting_graph.edge_index)
print(starting_graph)

for key, item in starting_graph:
    print(f'{key} found in data')

graph_info(starting_graph)

g = torch_geometric.utils.to_networkx(starting_graph, to_undirected=False)
nx.draw(g)
plt.show()
