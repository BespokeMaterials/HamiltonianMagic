"""
Utility functions for managing files and plots.
"""

import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import glob


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
    max_val = np.max(matrix)

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
