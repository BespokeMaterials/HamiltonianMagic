"""
This file will generate a dataset of chain graphs
  -@-@-@-@-@-@-@-
with nodes of different colors ()

The hamiltonian is build by following the next rules:
onsite  h_ii=c_i*0.5
hopping h_ij=h_ji=(c_i+c_j)*0.7 if ij is a edge

!!However in this dataset th ey will not be the hamiltonian but the DOS.!!
"""

import torch
import torch_geometric
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from obth_gnn.data import MaterialGraph
from obth_gnn.cost_functions.classical_denity_of_states import get_dos


class MyTensor(torch.Tensor):
    """
    this class is needed to work with graphs without edges
    """

    def max(self, *args, **kwargs):
        if torch.numel(self) == 0:
            return 0
        else:
            return torch.max(self, *args, **kwargs)


def create_circle_graph_dos(nr_nodes, colors=[1, 2], features_node=2, features_edge=3, steps=100, delta=0.1):
    """
    Constricts a random circular graph
    :param nr_nodes: (int) nr of nodes
    :param features_node: () nr of features/node
    :return:
    """
    # hamiltonian will be use later
    ham = torch.eye(nr_nodes)

    # Graph
    # Node features
    nf = [random.randint(-100, +100) for _ in range(features_node)]
    x = torch.tensor([nf for _ in range(nr_nodes)], dtype=torch.float)
    for i in range(nr_nodes):
        x[i][0] = random.choice(colors)
        ham[i][i] = 0.5 * x[i][0]

    # Edge indices
    ei = [i for i in range(nr_nodes)]  # 0,1,2,3,4
    ef = [i + 1 for i in range(nr_nodes)]  # 1,2,3,0
    ef[-1] = 0
    edges = [[*ei, *ef], [*ef, *ei]]
    edge_index = torch.tensor(edges, dtype=torch.long)

    # Edge features
    edge_attr = []
    for k, i in enumerate(edge_index[0]):
        ea = [random.randint(-100, +100) for _ in range(features_edge)]
        ea[0] = 1 + 0.5 * x[i][0] + 0.3 * x[edge_index[1][k]][0]
        edge_attr.append(ea)

    edge_attr = torch.tensor(edge_attr)

    # Global features
    u = torch.tensor([0.9, 1.0], dtype=torch.float)

    # Create Hamiltonian
    for k, i in enumerate(edge_index[0]):
        ham[i][edge_index[1][k]] = 0.7 * (x[i][0] + x[edge_index[1][k]][0])

    # Compute density of states
    dos = get_dos(ham, steps, delta)

    # Create custom graph
    graph = MaterialGraph(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u,
                          bond_batch=MyTensor(np.zeros(edge_index.shape[1])).long(), y=dos, dos0=dos[0], dos1=dos[1],
                          ham=ham)

    return graph, dos


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


# Build a dataset
class ColorChain(torch.utils.data.Dataset):
    def __init__(self, num_samples,
                 nr_nodes,
                 colors=[1, 2],
                 features_node=2,
                 features_edge=3):
        """
        Construct a custom dataset of colored circled graphs
        :param num_samples: (int) nr of the parameters in the data set.
        :param nr_nodes:(int)  nr on the nodes from the chain
        :param colors: (list) list of floats where etch values is considered a color
        :param features_node: (int) nr of features in the node there are dummy
        excepting the first one witch is the color the rest are dummy
        :param features_edge: (int) similar with features_node
        """

        self.num_samples = num_samples
        self.data_list = []
        for _ in range(num_samples):
            graph, dos = create_circle_graph_dos(nr_nodes, colors, features_node, features_edge)
            self.data_list.append((graph, dos))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_list[idx]


def main():
    """
    Example
    """
    # One graph example
    starting_graph, ham = create_circle_graph_dos(10, colors=[1])

    print("starting_graph", starting_graph.x)
    print("data.edge_index", starting_graph.edge_index)
    print("DOS:", starting_graph.y)
    print(starting_graph)
    # Plot density of states
    plt.plot(starting_graph.y[0], starting_graph.y[1])
    plt.xlabel('Energy')
    plt.ylabel('Density of States')
    plt.title('Density of States vs Energy')
    plt.savefig("img/DOS_line_nodes_10_color_1_2_dos.jpg")
    plt.show()

    for key, item in starting_graph:
        print(f'{key} found in data')

    graph_info(starting_graph)
    print(ham)

    g = torch_geometric.utils.to_networkx(starting_graph, to_undirected=False)
    nx.draw(g)
    plt.show()

    dataset = ColorChain(num_samples=100,
                         nr_nodes=10,
                         colors=[1, 2],
                         features_node=2,
                         features_edge=3)

    torch.save(dataset, 'artificial_graph_database/line_nodes_10_color_1_2_dos.pt')
    print("Done the dataset is constructed and saved")


if __name__ == "__main__":
    main()
