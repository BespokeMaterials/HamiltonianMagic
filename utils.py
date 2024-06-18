"""
Utility functions for managing files and plots.
"""

import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import glob
import math
import torch

## Save spot  ##
def save_spot(exp_name, spot_nr, model, data):
    # Create directory
    create_directory_if_not_exists("EXPERIMENTS")
    create_directory_if_not_exists(f"EXPERIMENTS/{exp_name}")
    create_directory_if_not_exists(f"EXPERIMENTS/{exp_name}/spot{spot_nr}")
    path = f"EXPERIMENTS/{exp_name}/spot{spot_nr}"
    path_img = os.path.join(path, "img")
    create_directory_if_not_exists(path_img)
    path_txt = os.path.join(path, "txt")
    create_directory_if_not_exists(path_txt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Save model
    torch.save(model.state_dict(), f"{path}/model.pt")

    # Save data
    for ko, inputs in enumerate(data):

        inputs = inputs.to(device)
        x = inputs.x.to(torch.float32)
        edge_index = inputs.edge_index.to(torch.int64)
        edge_attr = inputs.edge_attr.to(torch.float32)
        state = inputs.u.to(torch.float32)
        batch = inputs.batch
        bond_batch = inputs.bond_batch

        with torch.no_grad():
            hii, hij, ij = model(x, edge_index, edge_attr, state, batch, bond_batch)

        # Move tensors to CPU for further processing and numpy conversion
        hii = hii.cpu()
        hij = hij.cpu()
        ij = ij.cpu()

        pred_mat_r = torch.zeros([len(hii), len(hii)])
        pred_mat_i = torch.zeros([len(hii), len(hii)])
        for i, hi in enumerate(hii):
            pred_mat_r[i][i] = hi[0]
            pred_mat_i[i][i] = hi[1]

        for i, hx in enumerate(hij):
            pred_mat_r[ij[0][i]][ij[1][i]] = hx[0]
            pred_mat_i[ij[0][i]][ij[1][i]] = hx[1]

        target_mat_r = torch.zeros([len(hii), len(hii)])
        target_mat_i = torch.zeros([len(hii), len(hii)])
        for i, hi in enumerate(inputs.onsite):
            target_mat_r[i][i] = hi[0]
            target_mat_i[i][i] = hi[1]
        for i, hx in enumerate(inputs.hop):
            target_mat_r[ij[0][i]][ij[1][i]] = hx[0]
            target_mat_i[ij[0][i]][ij[1][i]] = hx[1]

        dif_mat_i = target_mat_i - pred_mat_i
        dif_mat_r = target_mat_r - pred_mat_r



        target_mat_r = target_mat_r.detach().numpy()
        pred_mat_r = pred_mat_r.detach().numpy()
        dif_mat_r = dif_mat_r.detach().numpy()
        dif_mat_i = dif_mat_i.detach().numpy()
        pred_mat_i=pred_mat_i.detach().numpy()
        target_mat_i=target_mat_i.detach().numpy()
        generate_heatmap(target_mat_r, filename=f'{path_img}/{ko}_tar_hmat.png')
        generate_heatmap(pred_mat_r, filename=f'{path_img}/{ko}_pred_hmat.png')
        generate_heatmap(dif_mat_r, filename=f'{path_img}/{ko}_dif_hmat.png')

        generate_heatmap(dif_mat_i, filename=f'{path_img}/{ko}_dif_smat.png')
        generate_heatmap(pred_mat_i, filename=f'{path_img}/{ko}_pred_smat.png')
        generate_heatmap(target_mat_i, filename=f'{path_img}/{ko}_target_smat.png')

        print("Done")
        print("max:", dif_mat_r.max())
        print("min:", dif_mat_r.min())


        np.save(os.path.join(path_txt, f'{ko}_dif_mat_hmat.npy'), dif_mat_r)
        np.save(os.path.join(path_txt, f'{ko}_target_mat_hmat.npy'), target_mat_r)
        np.save(os.path.join(path_txt, f'{ko}_pred_mat_hmat.npy'), pred_mat_r)

        np.save(os.path.join(path_txt, f'{ko}_dif_mat_smat.npy'), dif_mat_i)
        np.save(os.path.join(path_txt, f'{ko}_target_mat_smat.npy'), target_mat_i)
        np.save(os.path.join(path_txt, f'{ko}_pred_mat_smat.npy'), pred_mat_i)

        print("Done")



def nan_checker(lst):
    """
    Check if there are any NaN values in the list.

    Parameters:
    lst (list): The list to check for NaN values.

    Returns:
    bool: True if there is at least one NaN value in the list, False otherwise.
    """
    return any(math.isnan(x) for x in lst)

def generate_heatmap(matrix, filename, grid1_step=1, grid2_step=13):
    """
    Generate and save a heatmap from a given matrix.

    :param matrix: 2D array of data
    :param filename: The file name to save the heatmap
    :param grid1_step: Step for the first grid (default is 1)
    :param grid2_step: Step for the second grid (default is 13)
    """
    # Determine the min and max values of the matrix
    min_val = np.min(matrix)
    if min_val >=0:
        min_val=-0.1
    max_val = np.max(matrix)
    if max_val <=0:
        max_val=+0.1

    # Create the heatmap
    plt.figure()
    norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
    plt.imshow(matrix, cmap='seismic', norm=norm, interpolation='nearest')
    plt.colorbar()

    # Add grids
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], grid1_step), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], grid1_step), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.05)

    if grid2_step > 0:
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], grid2_step), minor=False)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], grid2_step), minor=False)
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.25)

    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def list_files_in_directory(directory_path):
    """
    List all files in the specified directory.

    :param directory_path: Path to the directory
    :return: List of file names in the directory
    """
    try:
        # Get the list of all files and directories
        entries = os.listdir(directory_path)

        # Filter out only the files
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]

        return files

    except FileNotFoundError:
        return f"The directory {directory_path} does not exist."
    except PermissionError:
        return f"Permission denied for accessing the directory {directory_path}."
    except Exception as e:
        return f"An error occurred: {e}"


def list_subdirectories(directory_path):
    """
    List all subdirectories in the specified directory.

    :param directory_path: Path to the directory
    :return: List of subdirectory names in the directory
    """
    try:
        # Get the list of all files and directories
        entries = os.listdir(directory_path)

        # Filter out only the subdirectories
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]

        return subdirectories

    except FileNotFoundError:
        return f"The directory {directory_path} does not exist."
    except PermissionError:
        return f"Permission denied for accessing the directory {directory_path}."
    except Exception as e:
        return f"An error occurred: {e}"


def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not exist.

    :param directory_path: Path to the directory to be created
    """
    try:
        # Check if the directory exists
        if not os.path.exists(directory_path):
            # Create the directory
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        else:
            print(f"Directory '{directory_path}' already exists.")
    except PermissionError:
        print(f"Permission denied for creating the directory '{directory_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def save_dict_to_json(dictionary, file_path):
    """
    Save a dictionary to a JSON file.

    :param dictionary: Dictionary to save
    :param file_path: Path to the JSON file
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)
        print(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def erase_png_files(directory):
    """
    Erases all .png files from the specified directory.

    Parameters:
    directory (str): The path to the directory where .png files should be erased.

    Returns:
    int: The number of .png files deleted.
    """
    # Construct the path to all .png files in the directory
    png_files = glob.glob(os.path.join(directory, '*.png'))

    # Delete each .png file
    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    return len(png_files)

def read_dict_from_json(file_path):
    """
    Reads a dictionary from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    dict: The dictionary read from the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None