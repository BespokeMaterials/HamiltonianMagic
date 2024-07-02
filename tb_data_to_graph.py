"""
Converts the JSON to a graph
"""

import numpy as np
import torch as tr

import math
from torch_geometric.data import Data
from scipy.special import sph_harm
from mendeleev import element
from tqdm import tqdm
from utils import list_files_in_directory, create_directory_if_not_exists, read_dict_from_json, nan_checker


## Fundamental graph elements and transformations ##
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


def element_to_atomic_number(element_symbol):
    try:
        el = element(element_symbol)
        return el.atomic_number
    except KeyError:
        return None  # Return None if the element is not found


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


## End: Fundamental graph elements and transformations ##


def get_nodes_from_structure(structure):
    # Construct the nodes
    node_features = []
    node_target = []
    col = 0
    for atom in structure["structure"]["atoms"]:

        # atomic number
        for orbit in range(atom["nr_orbitals"]):
            nod = []

            atomic_number = [element_to_atomic_number(atom["simbol"])]

            nod.extend(atomic_number)
            nod.extend([orbit])


            # position-> kils equivariance
            # position = atom["position"]
            # nod_s.extend(position)
            # nod_px.extend(position)
            # nod_py.extend(position)
            # nod_pz.extend(position)

            # onsite
            onsite = [structure["hmat"][col][col] * 10, structure["smat"][col][col] * 10]
            col += 1

            node_target.append(onsite)
            node_features.append(nod)

    node_features = tr.tensor(node_features, dtype=tr.float32)
    node_target = tr.tensor(node_target, dtype=tr.float32)

    return node_features, node_target


def get_edges_from_structure(structure, max_r=10):
    # Construct edges:
    edge_index = [[], []]
    edge_props = []
    edge_target = []

    # Extend atoms to orbitals
    # TODO: This is snot efficient change it:
    ext_coordinates = []
    ext_atom_type = []
    ext_orbitals = []
    for atom in structure["structure"]["atoms"]:
        for i in range(atom["nr_orbitals"]):
            ext_coordinates.append(atom["xyz"])
            ext_atom_type.append(element_to_atomic_number(atom["simbol"]))
            ext_orbitals.append(i)


    edges = structure["conections"]
    # Maybe add some diference

    for edge in edges:
        if edge[0] != edge[1]:
            edge_prop = []
            a = edge[0]
            b = edge[1]
            edge_index[0].append(a)
            edge_index[1].append(b)

            coord_a = tr.tensor(ext_coordinates[a])
            coord_b = tr.tensor(ext_coordinates[b])

            # print("ca",coord_a)
            distance = [tr.linalg.norm(coord_a-coord_b)]

            if distance[0]!=0:
                bassel_distance = bessel_distance(coord_a, coord_b, n=[i for i in range(1, 9)])
                spherical = spherical_harmonics(coord_a, coord_b,max_l=7)
                print("ok")
            else:
                bassel_distance=[0 for _ in range(8)]
                spherical = [0 for _ in range(42)]
                print("Zero ---")

            # print("distance:", distance)
            # print("bassel_distance:", len(bassel_distance))
            # print("spherical",len(spherical))
            # print("spherical", nan_checker(spherical))
            # print("bassel", nan_checker(bassel_distance))
            edge_prop.extend(distance)
            edge_prop.extend(bassel_distance)
            edge_prop.extend(spherical)
            # Add prop
            edge_props.append(edge_prop)

            # Target
            hopp = [structure["hmat"][a][b] * 10, structure["smat"][a][b] * 10]
            edge_target.append(hopp)

    # print(len(edge_props))
    edge_props = tr.tensor(edge_props, dtype=tr.float32)

    # print(len(edge_index[0]))
    # print(len(edge_index[1]))
    edge_index = tr.tensor(edge_index, dtype=tr.float32)
    edge_target = tr.tensor(edge_target, dtype=tr.float32)

    return edge_index, edge_props, edge_target


def get_global_from_structure(structure):
    # Global propriety:
    lattice_vectors = structure["structure"]['lattice vectors']
    print("lat vectors:", lattice_vectors)
    atom_xyz = structure["structure"]["atoms"]
    global_prop = [len(atom_xyz),
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
    return global_prop


def structure_to_graph(structure, radius=100):
    node_features, node_target = get_nodes_from_structure(structure)
    edge_index, edge_props, edge_target = get_edges_from_structure(structure, radius)
    global_prop = get_global_from_structure(structure)

    # Create custom graph
    graph = MaterialMesh(x=node_features,
                         edge_index=edge_index,
                         edge_attr=edge_props,
                         u=global_prop,
                         bond_batch=MyTensor(np.zeros(edge_index.shape[1])).long(),
                         hop=edge_target,
                         onsite=node_target)

    print("graph:", graph)
    return graph


def main(files_path, test_ratio, saving_spot, radius):
    # Construct the saving spot
    create_directory_if_not_exists(saving_spot)

    # ge the files and shuffle them:
    files = list_files_in_directory(files_path)
    # files=files[:5]
    # shuffle

    # Extract structure and build the graph
    structures = [read_dict_from_json(f"{files_path}/{st}") for st in files]
    #structures = structures[:5]
    graphs = [structure_to_graph(structure, radius) for structure in tqdm(structures)]

    train_ds = MaterialDS(graphs[:int(1 - len(graphs) * test_ratio)])
    tr.save(train_ds, f'{saving_spot}/train.pt')
    test_ds = MaterialDS(graphs[1 - int(len(graphs) * test_ratio):])
    tr.save(test_ds, f'{saving_spot}/test.pt')
    return 0


if __name__ == "__main__":
    test_ratio = 0.2
    files_path = "DATA/TB/BN_TB_JSON"
    saving_spot= "DATA/TB/BN_TB_GRAPH"
    radius = 50
    main(files_path, test_ratio,saving_spot ,radius)
