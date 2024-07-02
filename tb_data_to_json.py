"""
Convert the data from the DFT to more friendly and readable format.
Is a parser for aiida.fdf files
"""
import numpy as np
from sisl import get_sile
from utils import (list_subdirectories,
                    list_files_in_directory,
                   create_directory_if_not_exists,
                   save_dict_to_json,
                   generate_heatmap)
from tqdm import tqdm
import re

import numpy as np

# Read file functions
def read_edges_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize an empty list to store the edges
    edges = []

    for line in lines:
        # Skip lines that are comments or empty
        if line.startswith('#') or not line.strip():
            continue

        # Split the line into two parts and convert them to integers
        node1, node2 = map(int, line.split())
        # Append the edge as a tuple to the edges list
        edges.append((node1, node2))

    return edges
def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize an empty list to store the matrix
    matrix = []

    for line in lines:
        # Split the line into numbers and convert them to floats
        row = list(map(float, line.split()))
        # Append the row to the matrix
        matrix.append(row)

    # Convert the list of lists into a NumPy array
    matrix = np.array(matrix)

    return matrix
def read_file(file_path):
    """
    Reads the content of a file.

    Args:
    file_path (str): The path to the file to be read.

    Returns:
    str: The content of the file as a string.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    return content


# parse file
def extract_atomic_symbols_and_coordinates(file_path):
    """
    Extracts atomic symbols and their coordinates from the file content.

    Args:
    file_path (str): The path to the file containing the atomic data.

    Returns:
    tuple: A tuple containing two lists - one with atomic symbols and one with coordinates.
    """
    file_content = read_file(file_path)

    # Regular expression to match lines with atomic data
    atomic_data_pattern = re.compile(r'([A-Za-z]+)\s+([\d\.\-E]+)\s+([\d\.\-E]+)\s+([\d\.\-E]+)')

    atomic_symbols = []
    coordinates = []

    for line in file_content.splitlines():
        match = atomic_data_pattern.match(line)
        if match:
            atomic_symbols.append(match.group(1))
            coordinates.append([float(match.group(2)), float(match.group(3)), float(match.group(4))])

    return atomic_symbols, coordinates

def extract_lattice_vectors(input_string):
    """
    Extracts the lattice vectors from the input string.

    Args:
    input_string (str): The input string containing the lattice vectors.

    Returns:
    list: A list of lists representing the lattice vectors.
    """
    # Use regex to find the Lattice values
    match = re.search(r'Lattice="([\d\s\.\-]+)"', input_string)
    if match:
        # Extract the lattice values as a string
        lattice_string = match.group(1)
        # Split the string into individual numbers and convert them to float
        lattice_values = [float(x) for x in lattice_string.split()]
        # Group the values into 3x3 lattice vectors
        lattice_vectors = [lattice_values[i:i+3] for i in range(0, len(lattice_values), 3)]
        return lattice_vectors
    else:
        return "Lattice vectors not found in the input string."

def extract_lattice_vectors_from_file(file_path):
    """
    Extracts the lattice vectors from a file.

    Args:
    file_path (str): The path to the file containing the lattice vectors.

    Returns:
    list: A list of lists representing the lattice vectors.
    """
    file_content = read_file(file_path)
    return extract_lattice_vectors(file_content)


#Construct json
def construct_json_from_fdf(fdf_path, d_path):
    """
    :param fdf_path: Path to the input file aida.fdf
    :return: A dictionary with structure information.
    """

    nr=fdf_path.split("/")[-1].split("_")[0]

    ham_file=f"{d_path}/{nr}_Ham"
    hop_file=f"{d_path}/{nr}_IndsHop"
    d_p=d_path.split("/")[-1]
    structure=f"DATA/BN_database/structures/{d_p}/{nr}.txt"



    lattice_vectors = extract_lattice_vectors_from_file(structure)
    atomic_symbols, coordinates = extract_atomic_symbols_and_coordinates(structure)

    print("atomic_symbols:", atomic_symbols)
    print("coordinates:", coordinates)

    hmat=read_matrix_from_file(ham_file)
    print("h:",hmat)
    edges=read_edges_from_file(hop_file)
    print("edges:", edges)

    atoms = []
    for i, simbol in enumerate(atomic_symbols):
        atom = {
            "simbol": str(simbol),
            "xyz": coordinates[i],
            "nr_orbitals": 4,
        }
        atoms.append(atom)

    data = {
        "structure": {
            "lattice vectors": lattice_vectors,
            "atoms": atoms
        },
        "hmat": hmat.tolist(),
        "smat": hmat.tolist(),

        "conections":edges,
    }
    return data


def parse_tb(dft_path, json_path):
    create_directory_if_not_exists(json_path)
    create_directory_if_not_exists(f"{json_path}_img")

    print("source",json_path)
    source=dft_path.split("/")[-1]

    files = list_files_in_directory(dft_path)

    # filter files:
    files =[ file  for file in files if "Ham" in file]
    for sample in tqdm(files):
        new_file_name=f"{source}_{sample}"

        #construct dictionary
        json_dc = construct_json_from_fdf(f"{dft_path}/{sample}", dft_path)

        #sve dictionary to json
        save_dict_to_json(json_dc, f"{json_path}/{new_file_name}.json")


        hmat = np.array(json_dc["hmat"])
        filename = (f"{json_path}_img/{sample}_hmat.png")
        generate_heatmap(hmat, filename, grid1_step=1, grid2_step=4)
        smat = np.array(json_dc["smat"])
        filename = f"{json_path}_img/{sample}_smat.png"
        generate_heatmap(smat, filename, grid1_step=1, grid2_step=4)


def main(dft_path, json_path):
    parse_tb(dft_path, json_path)

    print("Files processed successfully!")
    return 0


if __name__ == "__main__":
    path_to_dft_files = "/home/ICN2/atomut/HamiltonianMagic/DATA/TB/Hams_noPBC/aBN"
    path_json = "DATA/TB/BN_TB_JSON"
    main(path_to_dft_files, path_json)
