"""
Convert the data from the DFT to more friendly and readable format.
Is a parser for aiida.fdf files
"""
from sisl import get_sile
from utils import (list_subdirectories,
                   create_directory_if_not_exists,
                   save_dict_to_json)
from tqdm import tqdm


def parse_atomic_file(file_path):
    """
    Parse the given file to extract atomic coordinates, atomic symbols, atomic types, and mesh cutoff.

    :param file_path: Path to the input file
    :return: A dictionary with atomic coordinates, symbols, types, and mesh cutoff
    """
    atomic_data = {
        "atomic_coordinates": [],
        "atomic_symbols": [],
        "atomic_types": [],
        "mesh_cutoff": None
    }

    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_species_block = False
    in_lattice_vectors_block = False
    in_atomic_coords_block = False

    species_dict = {}

    for line in lines:
        line = line.strip()

        if line.startswith("meshcutoff"):
            atomic_data["mesh_cutoff"] = line.split()[1] + " " + line.split()[2]

        if line == "%block chemicalspecieslabel":
            in_species_block = True
            continue

        if line == "%endblock chemicalspecieslabel":
            in_species_block = False
            continue

        if line == "%block lattice-vectors":
            in_lattice_vectors_block = True
            continue

        if line == "%endblock lattice-vectors":
            in_lattice_vectors_block = False
            continue

        if line == "%block atomiccoordinatesandatomicspecies":
            in_atomic_coords_block = True
            continue

        if line == "%endblock atomiccoordinatesandatomicspecies":
            in_atomic_coords_block = False
            continue

        if in_species_block:
            parts = line.split()
            species_dict[int(parts[0])] = parts[2]

        if in_atomic_coords_block:
            parts = line.split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            atom_type = int(parts[3])
            atomic_symbol = species_dict[atom_type]
            atomic_data["atomic_coordinates"].append((x, y, z))
            atomic_data["atomic_symbols"].append(atomic_symbol)
            atomic_data["atomic_types"].append(atom_type)

    return atomic_data


def construct_json_from_fdf(fdf_path):
    print(fdf_path)
    fdf = get_sile(fdf_path)
    h = fdf.read_hamiltonian()
    hmat = h.Hk([0, 0, 0]).todense()
    smat = h.Sk([0, 0, 0]).todense()
    geometry = fdf.read_geometry()
    lattice_vectors = geometry.cell
    atomic_symbols = geometry.atoms

    atomic_info = parse_atomic_file(fdf_path)
    atoms = []
    for i, simbol in enumerate(atomic_info["atomic_symbols"]):
        atom = {
            "simbol": simbol,
            "xyz": atomic_info['atomic_coordinates'][i],
            "nr_orbitals": atomic_symbols[atomic_info['atomic_types'][i] - 1],
        }
        atoms.append(atom)

    data = {
        "structure": {
            "lattice vectors": lattice_vectors,
            "atoms": atoms
        },
        "hmat": hmat,
        "smat": smat
    }
    return data


def parse_dft(dft_path, json_path):
    create_directory_if_not_exists(json_path)

    dft_reg = list_subdirectories(dft_path)
    for sample in tqdm(dft_reg):
        path = f"{dft_path}/{sample}/aiida.fdf"
        json_dc = construct_json_from_fdf(path)
        save_dict_to_json(json_dc, f"{json_path}/{sample}.json")ç




def main(dft_path, json_path):
    parse_dft(dft_path, json_path)

    print("Files processed successfully!")
    return 0


if __name__ == "__main__":
    path_to_dft_files = "DATA/DFT/BN_DFT"
    path_json = "DATA/DFT/BN_DFT_JSON"
    main(path_to_dft_files, path_json)
