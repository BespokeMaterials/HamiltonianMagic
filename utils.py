"""
Utility functions for managing files and plots.
"""

import os
import json
import matplotlib.pyplot as plt


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




    