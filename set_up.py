import subprocess
import sys

def install_requirements(requirements_file):
    """
    Install packages listed in the given requirements file.
    """
    try:
        with open(requirements_file, 'r') as file:
            packages = file.readlines()
            for package in packages:
                package = package.strip()
                if package:
                    print(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All packages installed successfully.")
    except FileNotFoundError:
        print(f"File {requirements_file} not found.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing packages: {e}")

if __name__ == "__main__":
    requirements_file = 'requirements.txt'
    install_requirements(requirements_file)