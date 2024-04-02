"""
This code will take all the xyz file from a directory
and will construct dataset out of them that will have the following: Structure-onsite-hopping
"""

from simpletb.parser import struct_from_xyz
from simpletb.neighbors import get_neighbor_indexes
from simpletb import SiteList, System
import numpy as np
import torch as tr
import periodictable
import os
import math
from torch_geometric.data import Data, Dataset, InMemoryDataset
from copy import deepcopy
from scipy.special import sph_harm


class MaterialMesh(Data):
    def __init__(self, x, edge_index, edge_attr, u, bond_batch, hop, onsite):
        super(MaterialMesh, self).__init__()
        self.x = x  # Node features
        self.edge_index = edge_index  # Edge indices
        self.edge_attr = edge_attr  # Edge features
        self.u = u  # Global features

        self.bond_batch = bond_batch  # tels from witch batch is the edge
        self.onsite = onsite  # target propriety
        self.hop = hop  # target hopping

    def __cat_dim__(self, key, value, *args, **kwargs):
        """
        Ad extra dim when batched u.
        It will make then to not concatenate
        :param key:
        :param value:
        :param args:
        :param kwargs:
        :return:
        """
        if key == "u":
            return None

        return super().__cat_dim__(key, value, *args, **kwargs)


def get_atomic_number(symbol):
    try:
        element = periodictable.elements.symbol(symbol)
        atomic_number = element.number
        return atomic_number
    except AttributeError:
        return None


def onsite_to_dict(onsite_list):
    os = {}
    for o in onsite_list:
        os[o[0]] = o[1]
    return os


class MyTensor(tr.Tensor):
    """
    this class is needed to work with graphs without edges
    """

    def max(self, *args, **kwargs):
        if tr.numel(self) == 0:
            return 0
        else:
            return tr.max(self, *args, **kwargs)


def hoping_to_dict(hoping_list):
    ho = {}
    for h in hoping_list:
        ho[h[0]] = {}

    for h in hoping_list:
        ho[h[0]][h[1]] = 10+h[2] #TODO do this +10 properly
    return ho


def xyz_to_system(xyz_file):
    """

    :param xyz_file:
    :return: material as a System
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


def f_cut(r, decay_rate=3, cutoff=0.5):
    """
    Computes the cosine decay cutoff function.

    Parameters:
        r (float or numpy array): Distance value(s).
        decay_rate (float): Decay rate parameter.

    Returns:
        float or numpy array: Output value(s) of the cosine decay cutoff function.
    """
    #return 0.5 * (1 + np.cos(np.pi * r)) * np.exp(-decay_rate * r)
    # Compute values of cutoff function
    cutoffs = 0.5 * (np.cos(r * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    cutoffs *= (r < cutoff)
    return cutoffs




def bessel_distance(c1, c2, n=[1, 2, 3, 4, 5, 6], rc=3):
    # print(f"c1:{c1}, c2:{c2}")
    d = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2
    rij = np.sqrt(d * d)
    c=np.sqrt(2 / rc)
    fc =f_cut(rij, rc * 0.5)
    bes = [c * fc* (np.sin(n_ * math.pi * rij / rc)) / rij for n_ in n]

    return bes


def spherical_harmonics(c1, c2,max_l=1 ):
    # muve to center
    rc=c1-c2
    r, theta, phi = cartesian_to_spherical(rc[0], rc[1], rc[2])
    y=[]
    for l in range(max_l):
        # yl=[]
        for m in range(-l, l):
            ylm=real_spherical_harmonics(l, m, theta, phi)
            y.append(ylm)
        # y.append(yl)
    return y


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def real_spherical_harmonics(l, m, theta, phi):
    # Compute the complex spherical harmonics
    Y_lm_complex = sph_harm(m, l, phi, theta)

    # Compute real spherical harmonics based on m value
    if m > 0:
        return np.sqrt(2) * np.real(Y_lm_complex)
    elif m == 0:
        return np.real(Y_lm_complex)
    else:
        return np.sqrt(2) * (-1) ** m * np.imag(Y_lm_complex)

def xyz_to_graph(xyz_file):
    """
    Convert xyz_file toa graph.
    :param xyz_file:
    :return: gpytorch geometric graph.
    """

    material = xyz_to_system(xyz_file)
    neighbours = material.get_neighbours()
    onsite = onsite_to_dict(material.get_onsite())
    hopping = hoping_to_dict(material.get_hopping())
    site_list = material.site_list

    # Construct the nodes
    node_features = []
    node_target = []
    print("sl:", len(site_list.site_list))
    for site in site_list.site_list:
        atomic_number = [get_atomic_number(site.label.split("_")[0])]
        orbital = [1]  # we are working only with one orbital  now
        position = [x for x in site.coord]
        node_f = []
        node_f.extend(atomic_number)
        node_f.extend(orbital)
        node_f.extend(position)
        node_features.append(node_f)
        on = [onsite[site.uid].real, onsite[site.uid].imag]
        node_target.append(on)
    node_features = tr.tensor(deepcopy(node_features), dtype=tr.float32)
    node_target = tr.tensor(node_target, dtype=tr.float32)
    print("len nf", len(node_features))
    print("len nt", len(node_target))

    # Construct edges:
    edge_index = [[], []]
    edge_props = []
    edge_target = []
    for n_ in neighbours:
        na = n_[0]
        for nb in n_[1]:
            if nb != na:
                edge_prop = []
                edge_index[0].append(na)
                edge_index[1].append(nb)
                na_s = site_list.get_sites([na])[0]
                nb_s = site_list.get_sites([nb])[0]

                distance = [np.sqrt(sum([s ** 2 for s in na_s.get_coord() - nb_s.get_coord()]))]
                # print("distance:", distance)
                # Bassel
                # TODO:
                bassel_distance = bessel_distance(na_s.get_coord(), nb_s.get_coord(), n=[i for i in range(1,9)])
                # Spherical
                spherical =spherical_harmonics(na_s.get_coord(), nb_s.get_coord(),max_l=4 ) #spherical_harmonics(na_s.get_coord(), na_s.get_coord())

                edge_prop.extend(distance)
                edge_prop.extend(bassel_distance)
                edge_prop.extend(spherical)

                # print(hopping)
                hopp = [hopping[na][nb].real, hopping[na][nb].imag]
                edge_target.append(hopp)
                edge_props.append(edge_prop)
    print(len(edge_props))
    edge_props = tr.tensor(edge_props, dtype=tr.float32)

    print(len(edge_index[0]))
    print(len(edge_index[1]))
    edge_index = tr.tensor(edge_index, dtype=tr.float32)
    edge_target = tr.tensor(edge_target, dtype=tr.float32)

    # Global propriety:
    global_prop = [len(site_list.site_list)]
    global_prop = tr.tensor(global_prop)

    print("AICI")
    print("len nf", len(node_features))
    # Create custom graph
    graph = MaterialMesh(x=node_features,
                         edge_index=edge_index,
                         edge_attr=edge_props,
                         u=global_prop,
                         bond_batch=MyTensor(np.zeros(edge_index.shape[1])).long(),
                         hop=edge_target,
                         onsite=node_target)
    return graph


# Build a dataset
class MaterialDS(tr.utils.data.Dataset):
    def __init__(self, graph_list):
        """
        Convert a list  of graphs into a dataset.
        :param graph_list: [list of pytorch geometric graphs]
        """
        # (g.onsite, g.hop)
        self.data_list = [(g) for g in graph_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def main():
    input_xyz_directory = "artificial_graph_database/DummyGraphene/substitution_2"
    # Extract all the files from directory:
    files = [f for f in os.listdir(input_xyz_directory) if os.path.isfile(os.path.join(input_xyz_directory, f))]

    graphs = []
    for file in files:
        graph = xyz_to_graph(f"{input_xyz_directory}/{file}")
        graphs.append(graph)

    print(graphs[0])
    material_ds = MaterialDS(graphs)
    tr.save(material_ds, 'artificial_graph_database/DummyGrapheneGraph/graphene_dm_02.pt')
    return 0


if __name__ == "__main__":
    main()
