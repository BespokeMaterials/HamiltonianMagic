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
from tqdm import tqdm

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

class MyTensor(tr.Tensor):
    """
    this class is needed to work with graphs without edges
    """

    def max(self, *args, **kwargs):
        if tr.numel(self) == 0:
            return 0
        else:
            return tr.max(self, *args, **kwargs)


def read_file_line_by_line(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append([line.strip()])  # Remove trailing newline characters
    return (lines)


def n_parser(n_file):
    n_lines = read_file_line_by_line(n_file)
    n_lines = [[int(l)-1 for l in line[0].split()] for line in n_lines]

    return n_lines


def s_parser(s_file):
    s_lines = read_file_line_by_line(s_file)
    s_lines = [[float(l) for l in line[0].split()] for line in s_lines]
    return s_lines


def e_parser(e_file):
    s_lines = read_file_line_by_line(e_file)
    s_lines = [[float(l) for l in line[0].split()] for line in s_lines]
    return s_lines


def read_xyz_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lattice_vectors = [[float(val) for val in lines[i].strip().split()] for i in range(2)]
        lv = [[lattice_vectors[0][0], lattice_vectors[0][1], 0],
              [lattice_vectors[1][0], lattice_vectors[1][1], 0],
              [00.0, 0.0, 1.0]]
        acc = float(lines[2])
        cut_acc = float(lines[3])
        num_atoms = int(lines[4])
        atom_xyz = [[float(val) for val in line.strip().split()] for line in lines[5:]]
    return lv, acc, cut_acc, atom_xyz


def f_cut(r, decay_rate=3, cutoff=0.5):
    """
    Computes the cosine decay cutoff function.

    Parameters:
        r (float or numpy array): Distance value(s).
        decay_rate (float): Decay rate parameter.

    Returns:
        float or numpy array: Output value(s) of the cosine decay cutoff function.
    """
    # return 0.5 * (1 + np.cos(np.pi * r)) * np.exp(-decay_rate * r)
    # Compute values of cutoff function
    cutoffs = 0.5 * (np.cos(r * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    cutoffs *= (r < cutoff)
    return cutoffs


def bessel_distance(c1, c2, n=[1, 2, 3, 4, 5, 6], rc=3):
    # print(f"c1:{c1}, c2:{c2}")
    d = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2
    rij = np.sqrt(d * d)
    c = np.sqrt(2 / rc)
    fc = f_cut(rij, rc * 0.5)
    bes = [c * fc * (np.sin(n_ * math.pi * rij / rc)) / rij for n_ in n]

    return bes


def spherical_harmonics(c1, c2, max_l=1):
    # muve to center
    rc = c1 - c2
    r, theta, phi = cartesian_to_spherical(rc[0], rc[1], rc[2])
    y = []
    for l in range(max_l):
        # yl=[]
        for m in range(-l, l):
            ylm = real_spherical_harmonics(l, m, theta, phi)
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


def to_graph(n, nn, s, ss, e, lattice_vectors, acc, cut_acc, atom_xyz):
    # Construct the nodes
    node_features = []
    node_target = []

    for i, site in enumerate(atom_xyz):
        atomic_number = [6]
        orbital = [1]  # we are working only with one orbital  now
        position = [x for x in site]
        print(position)
        node_f = []
        node_f.extend(atomic_number)
        node_f.extend(orbital)
        node_f.extend(position)
        node_features.append(node_f)
        on = [e[i][0]*1000, 0*1000]
        node_target.append(on)

    # print("node f:", node_features)
    # print("node f", e)
    node_features = tr.tensor(node_features, dtype=tr.float32)
    node_target = tr.tensor(node_target, dtype=tr.float32)
    print("len nf", len(node_features))
    print("len nt", len(node_target))

    # Construct edges:
    edge_index = [[], []]
    edge_props = []
    edge_target = []

    for na, n_i in tqdm(enumerate(n)):
        for j, nb in enumerate(n_i):
            if nb != na:
                edge_prop = []
                edge_index[0].append(na)
                edge_index[1].append(nb)
                na_s_coord = tr.tensor(atom_xyz[na])
                # print(nb)
                nb_s_coord = tr.tensor(atom_xyz[nb])
                if na <0 or nb <0:
                    print("oooooo")
                # print(na_s_coord)
                distance = [np.sqrt(sum([s ** 2 for s in na_s_coord - nb_s_coord]))]
                # print("distance:", distance)
                # Bassel
                # TODO:
                bassel_distance = bessel_distance(na_s_coord, nb_s_coord, n=[i for i in range(1, 9)],
                                                  rc=acc * cut_acc + 0.1)
                # Spherical
                spherical = spherical_harmonics(na_s_coord, nb_s_coord,
                                                max_l=4)  # spherical_harmonics(na_s.get_coord(), na_s.get_coord())

                edge_prop.extend(distance)
                edge_prop.extend(bassel_distance)
                edge_prop.extend(spherical)

                # print(hopping)
                hopp = [s[na][j]*1000, 0*1000]
                edge_target.append(hopp)
                edge_props.append(edge_prop)

    for na, n_i in tqdm(enumerate(nn)):
        for j, nb in enumerate(n_i):
            if nb != na:
                edge_prop = []
                edge_index[0].append(na)
                edge_index[1].append(nb)


                na_s_coord = tr.tensor(atom_xyz[na])
                # print(nb)
                nb_s_coord = tr.tensor(atom_xyz[nb])
                distance = [np.sqrt(sum([s ** 2 for s in na_s_coord - nb_s_coord]))]
                # print("distance:", distance)
                # Bassel
                # TODO:
                bassel_distance = bessel_distance(na_s_coord, nb_s_coord, n=[i for i in range(1, 9)],
                                                  rc=acc * cut_acc + 0.1)
                # Spherical
                spherical = spherical_harmonics(na_s_coord, nb_s_coord,
                                                max_l=4)  # spherical_harmonics(na_s.get_coord(), na_s.get_coord())

                edge_prop.extend(distance)
                edge_prop.extend(bassel_distance)
                edge_prop.extend(spherical)

                # print(hopping)
                hopp = [ss[na][j]*1000, 0*1000]
                edge_target.append(hopp)
                edge_props.append(edge_prop)

    print(len(edge_props))
    edge_props = tr.tensor(edge_props, dtype=tr.float32)

    print(len(edge_index[0]))
    print(len(edge_index[1]))
    edge_index = tr.tensor(edge_index, dtype=tr.float32)
    edge_target = tr.tensor(edge_target, dtype=tr.float32)

    # Global propriety:
    global_prop = [len(atom_xyz),
                   acc, cut_acc,
                   lattice_vectors[0][0],
                   lattice_vectors[0][1],
                   lattice_vectors[0][2],
                   lattice_vectors[1][0],
                   lattice_vectors[1][1],
                   lattice_vectors[1][2],
                   lattice_vectors[2][0],
                   lattice_vectors[2][1],
                   lattice_vectors[2][2]]

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
    print("graph",graph)
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


def file_to_grph( file_d, g_name,xyz):


    n_file = f"Samples_for_Andrei/{file_d}/n.txt"
    nn_file = f"Samples_for_Andrei/{file_d}/nn.txt"
    s_file = f"Samples_for_Andrei/{file_d}/s.txt"
    ss_file = f"Samples_for_Andrei/{file_d}/ss.txt"
    e_file = f"Samples_for_Andrei/{file_d}/e.txt"
    xyz_file = f"Samples_for_Andrei/{file_d}/{xyz}"
    n = n_parser(n_file)
    nn = n_parser(nn_file)
    s = s_parser(s_file)
    ss = s_parser(ss_file)
    e = e_parser(e_file)
    read_xyz_file(xyz_file)
    lattice_vectors, acc, cut_acc, atom_xyz = read_xyz_file(xyz_file)
    print("lattice vectors:", lattice_vectors)
    print("acc:", acc)
    print("cut_acc:", cut_acc)

    graph18 = to_graph(n, nn, s, ss, e, lattice_vectors, acc, cut_acc, atom_xyz)
    graphs = [graph18]
    material_ds = MaterialDS(graphs)
    tr.save(material_ds, f'Aron/{g_name}')
    return graph18

def main():

    graphs=[]
    print("13")
    g= file_to_grph( file_d="13-nm", g_name="13-nm.pt",xyz="sample_13nm_1.xyz")
    graphs.append(g)
    print("18")
    g = file_to_grph(file_d="18-nm", g_name="18-nm.pt",xyz="sample_18nm_1.xyz")
    graphs.append(g)
    material_ds = MaterialDS(graphs)
    tr.save(material_ds, f'Aron/13-18-nn')
    print("25")
    g = file_to_grph(file_d="25-nm", g_name="25-nm.pt",xyz="sample_25.5nm_1.xyz")
    graphs.append(g)
    material_ds = MaterialDS(graphs)
    tr.save(material_ds, f'Aron/13-18-25-nn')

    print("Done")
    return 0


if __name__ == "__main__":
    main()