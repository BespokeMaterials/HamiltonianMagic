"""
Construct the graph for hBn and TB model
"""

import numpy as np
import torch as tr

import os
import math
from torch_geometric.data import Data
from scipy.special import sph_harm
from mendeleev import element

# some untility functions that should be encapsualted in the core model

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

# End utility function #

def list_files(directory):
    """Returns a list of all file names in the specified directory."""
    try:
        return os.listdir(directory)
    except FileNotFoundError:
        return "Directory not found."


def read_matrix_from_file(file_path):
    """Reads a numerical matrix from a file where each line is a row of the matrix."""
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line to a list of floats
            row = [float(num) for num in line.split()]
            matrix.append(row)
    return matrix


def read_lndsHop_file(file_path):
    """Reads a file containing pairs of indices and returns them as a matrix of rows and columns."""
    rows = []
    columns = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            if line.strip():  # Check if line is not empty
                parts = line.split()
                rows.append(int(parts[0]))
                columns.append(int(parts[1]))
    return [rows, columns]

def read_lattice_vector(filename):
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("Lattice="):
                # Extract the part of the string containing the lattice vector
                lattice_part = line.split("Lattice=\"")[1].split("\"")[0]
                # Split the extracted string into individual components and convert them to floats
                lattice_vector = np.array(list(map(float, lattice_part.split())))
                # Reshape the 1D vector into a 3x3 matrix
                lattice_matrix = lattice_vector.reshape((3, 3))

                return lattice_matrix

def read_xyz_file(file_path):
    """Reads an extended XYZ file and returns a dictionary with atoms, lattice vectors, and other properties."""
    data = {
        "atoms": [],
        "lattice_vectors": [],
        "energy": None,
        "pbc": None
    }

    data['lattice_vectors'] = read_lattice_vector(file_path)
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line (number of atoms)
        header = next(file).strip()

        # Extract information from the header
        parts = header.split()



        # Other properties like energy and pbc
        energy_index = next((i for i, part in enumerate(parts) if part.startswith('energy=')), None)
        if energy_index is not None:
            data['energy'] = float(parts[energy_index].split('=')[1])

        pbc_index = next((i for i, part in enumerate(parts) if part.startswith('pbc="')), None)
        if pbc_index is not None:
            data['pbc'] = parts[pbc_index].split('"')[1].split()

        # Read atom positions and properties
        for line in file:
            if line.strip():  # Ensure the line is not empty
                atom_data = line.split()
                atom = {
                    "element": atom_data[0],
                    "position": list(map(float, atom_data[1:4])),
                    "forces": list(map(float, atom_data[4:7]))
                }
                data['atoms'].append(atom)

    return data


def info_to_graph(info):

    # Construct the nodes
    node_features = []
    node_target = []
    col=0
    for  atom  in info["structure"]["atoms"]:

        nod_s =[]
        nod_px =[]
        nod_py =[]
        nod_pz=[]
        #atomic number

        atomic_number =  [element_to_atomic_number(atom["element"])]
        nod_s.extend(atomic_number)
        nod_px.extend(atomic_number)
        nod_py.extend(atomic_number)
        nod_pz.extend(atomic_number)
        # orbitals
        nod_s.extend([1])
        nod_px.extend([2])
        nod_py.extend([3])
        nod_pz.extend([4])
        # position
        position= atom["position"]
        nod_s.extend(position)
        nod_px.extend(position)
        nod_py.extend(position)
        nod_pz.extend(position)
        # forces
        forces=atom["forces"]
        nod_s.extend(forces)
        nod_px.extend(forces)
        nod_py.extend(forces)
        nod_pz.extend(forces)

        node_features.append(nod_s)
        node_features.append(nod_px)
        node_features.append(nod_py)
        node_features.append(nod_pz)


        #onsite

        on_s=[info["ham"][col][col]*10,0]
        on_px = [info["ham"][col+1][col+1]*10,0]
        on_py = [info["ham"][col+2][col+2]*10,0]
        on_pz = [info["ham"][col+3][col+3]*10,0]
        col+=4
        node_target.append(on_s)
        node_target.append(on_px)
        node_target.append(on_py)
        node_target.append(on_pz)


    node_features = tr.tensor(node_features, dtype=tr.float32)
    node_target = tr.tensor(node_target, dtype=tr.float32)
    # print("len nf", len(node_features))
    # print("len nt", len(node_target))


    # Construct edges:
    edge_index = [[], []]
    edge_props = []
    edge_target = []
    # print("ee:", len(info["edges"][0]))
    for i in range(len(info["edges"][0])):

        edge_prop=[]
        a=info["edges"][0][i]
        b=info["edges"][1][i]
        edge_index[0].append(a)
        edge_index[1].append(b)

        atom_a=info["structure"]["atoms"][int(a/4)]
        atom_b = info["structure"]["atoms"][int(b / 4)]
        coord_a=tr.tensor(atom_a["position"])
        coord_b=tr.tensor(atom_b["position"])
        forces_a=atom_a["forces"]
        forces_b=atom_b["forces"]

        # print("ca",coord_a)
        distance = [np.sqrt(sum([s ** 2 for s in coord_a - coord_b]))]
        bassel_distance = bessel_distance(coord_a, coord_b, n=[i for i in range(1, 9)])
        spherical = spherical_harmonics(coord_a, coord_b,max_l=7)  # spherical_harmonics(na_s.get_coord(), na_s.get_coord())

        # TODO: add forces if needed
        # Add forces

        edge_prop.extend(distance)
        edge_prop.extend(bassel_distance)
        edge_prop.extend(spherical)
        edge_props.append(edge_prop)



        # edge target
        # print(hopping)
        # print("ham shape:",len( info["ham"]), len(info["ham"][0]))
        # print("ii:",i)
        hopping = info["ham"][a][b]*10
        hopp = [hopping.real, hopping.imag]
        edge_target.append(hopp)


    print(len(edge_props))
    edge_props = tr.tensor(edge_props, dtype=tr.float32)

    print(len(edge_index[0]))
    print(len(edge_index[1]))
    edge_index = tr.tensor(edge_index, dtype=tr.float32)
    edge_target = tr.tensor(edge_target, dtype=tr.float32)



    # Global propriety:
    lattice_vectors = info["structure"]['lattice_vectors']
    print("lat vectors:", lattice_vectors)
    atom_xyz=info["structure"]["atoms"]
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

    # Create custom graph
    graph = MaterialMesh(x=node_features,
                         edge_index=edge_index,
                         edge_attr=edge_props,
                         u=global_prop,
                         bond_batch=MyTensor(np.zeros(edge_index.shape[1])).long(),
                         hop=edge_target,
                         onsite=node_target)
    print("graph", graph)
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
    dname = "aBN"
    directory_path = f'BN_database/Hams_noPBC/{dname}'
    xyz= f'BN_database/structures-20240403T095638Z-001/structures/{dname}'
    files = list_files(directory_path)
    print(files)

    infos = {}
    for file in files:
        nr = int(file.split("_")[0])
        name = f"{dname}_{nr}"
        if name not in infos.keys():
            infos[name] = {}


        tip = file.split("_")[1]
        if tip == "Ham":
            infos[name]["ham"] = read_matrix_from_file(f"{directory_path}/{file}")
        elif tip == "IndsHop":
            infos[name]["edges"] = read_lndsHop_file(f"{directory_path}/{file}")

        infos[name]["structure"] =read_xyz_file(f"{xyz}/{nr}.txt")
        print( infos[name]["structure"])

    graphs=[]
    for key in infos.keys():
        info=infos[key]
        graph=info_to_graph(info)
        graphs.append(graph)

    material_ds = MaterialDS(graphs)
    tr.save(material_ds, f'BN_database/Graphs/{dname}.pt')
    return 0





if __name__ == "__main__":
    main()