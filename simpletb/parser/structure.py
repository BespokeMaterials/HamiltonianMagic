"""
Read from xyz file
"""
from ase.cell import Cell
from ase.io import read

import numpy as np


def struct_from_xyz(filename):
    """

    Args:
        filename (str):  path theo the xyz file with the system information

    Returns: lattice vectors (as Column vectors), atom:types, atomic coordinates

    """
    atoms = read(filename, format='xyz')
    atom_types = atoms.get_chemical_symbols()
    coordinates = atoms.get_positions()
    lattice_vectors = parse_lattice_vectors(filename)

    return np.transpose(lattice_vectors), atom_types, coordinates


def parse_lattice_vectors(filename):
    """
     required for ase xyz files.
    Args:
        filename (str): path to the xyz file

    Returns:

    """
    with open(filename, 'r') as f:
        # Read the second line
        second_line = f.readlines()[1].strip()

        # Extract lattice vectors from the second line
        lattice_str = second_line.split('Lattice="')[1]
        lattice_str=lattice_str.split('" ')[0]
        lattice_values = [float(val) for val in lattice_str.split()]
        lattice_vectors = np.array(lattice_values).reshape(3, 3)

    return lattice_vectors