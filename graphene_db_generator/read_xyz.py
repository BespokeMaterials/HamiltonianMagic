"""
0
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import networkx as nx

def read_xyz(filename):
    """
    Read XYZ file using ASE library and return atomic positions.
    :param filename: (str) path to the xyz file.
    :return: atoms, positions
    """

    atoms = read(filename)
    positions = atoms.get_positions()
    return atoms, positions

def compute_bonds(positions, cutoff_distance=1.5):
    """
    Compute bonds between atoms based on distance.
    :param positions:
    :param cutoff_distance:
    :return: bonds as (i j) list
    """

    n_atoms = len(positions)
    bonds = []

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < cutoff_distance:
                bonds.append((i, j))

    return bonds

def construct_graph_nx(atoms,positions, bonds):
    """
    Construct a graph where atoms are nodes and bonds are edges.
    """
    G = nx.Graph()

    for i, atom in enumerate(atoms):
        G.add_node(i, element=atom.symbol,pos= positions[i])

    G.add_edges_from(bonds)

    return G

if __name__ == "__main__":
    # Replace 'your_file.xyz' with the path to your XYZ file
    filename = '/home/ICN2/atomut/Documents/GitHub/hBNandGNN/artificial_graph_database/Graphene/ag.xyz'

    # Read XYZ file
    atoms, positions = read_xyz(filename)

    # Compute bonds
    bonds = compute_bonds(positions)

    # Construct graph
    G = construct_graph_nx(atoms,positions, bonds)

    # Example: Print nodes and edges of the graph
    print("Nodes:", G.nodes(data=True))
    print("Edges:", G.edges())
    nx.draw(G)
    plt.show()