"""
This code will take all the xyz file from a directory
and will construct dataset out of them that will have the following: Structure-onsite-hopping
"""

from simpletb.parser import struct_from_xyz
from simpletb.neighbors import get_neighbor_indexes
from simpletb import SiteList
import numpy as np
import torch as tr

def xyz_to_system(xyz_file):
    """

    :param xyz_file:
    :return: material as an System
    """
    lattice_vectors, atom_types, coordinates = struct_from_xyz(xyz_file)
    site_list = SiteList(atom_types, coordinates)
    print("Nr of sites:", len(site_list.site_list))

    def hopping_calculator(system, site_a, site_b):
        """
        Computes the hoping between two sites.
        Args:
            system: system from witch siteA and siteB are extracted.
            site_a: site A
            site_b: Site B

        Returns: hopping value

        """
        t10 = -2.414
        t20 = -0.168

        r = np.linalg.norm(site_a.coord - site_b.coord)
        if r <= (1 + np.sqrt(3)) * 1.24 / 2:
            t = t10 * np.exp(1.847 * (r - 1.24))
        elif r <= (2 + np.sqrt(3)) * 1.24:
            t = t20 * np.exp(-0.168 * (r - 1.24 * np.sqrt(3)))
        else:
            t = 0.0

        return t + 0j

    def onsite_calculator(system, site_a):
        """
        Computes the onsite
        Args:
            system: system from witch siteA is extracted
            site_a: site A

        Returns:

        """

        simbol = site_a.label.split("_")[0]
        if simbol == "C":
            onsite = 2
        else:
            onsite = 3.0
        return onsite

        def get_neighbours(system):
            """
            Function to decide the neighbours
            Args:
                system: System in witch we need to compute the neighbours

            Returns: list of neighbours []

            """
            return get_neighbor_indexes(system, cutoff=3, pbc=(True, True, False))

        # Build the material:
        material = System(site_list, lattice_vectors)
        material.set_onsite_function(onsite_calculator)
        material.set_hopping_function(hopping_calculator)
        material.set_compute_neighbours(get_neighbours)

        return material
def xyz_to_graph( xyz_file):
    """
    Convert xyz_file toa graph.
    :param xyz_file:
    :return: gpytorch geometric graph.
    """

    material=xyz_to_system(xyz_file)
    neighbours=material.get_neighbours()
    onsite=material.get_onsite()
    hopping=material.get_hopping()
    site_list=material.site_list


    # Construct the nodes
    node_features = []
    node_target=[]
    for site in site_list.site_list:
        atomic_number = []
        orbital = [1]
        position = []
        node_f = []
        node_features.append(node_f)
        on=[]
        node_target.append(on)
    node_features=tr.tensor(node_features)
    node_target=tr.tensor(node_target)

    # Construct edges:
    edge_index=[[][]]
    edge_prop=[]
    for n_  in neighbours:
        na=n_[0]
        edge_index[0].append(na
        for nb in n_[1]:




def main():

    input_xyz_directory="artificial_graph_database/DummyGraphene/substitution"
    # Extract all the files from directory:
    files=[]
    graphs=[]
    for file in files:


        lattice_vectors, atom_types, coordinates = struct_from_xyz(xyz_file)

        site_list = SiteList(atom_types, coordinates)
        print("Nr of sites:", len(site_list.site_list))

        node_features = [random.randint(-100, +100) for _ in range(features_node)]
        # Node target

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

        # Edge target


    return  0
if __name__ == "__main__":
    main()