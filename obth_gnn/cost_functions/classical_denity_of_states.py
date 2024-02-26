"""
Computes the density of states in the classical way.
It does not scale wale for big system we recommend using LSquant.
"""

import torch

def soft_threshold(x, delta):
    """
    Required to not kill the gradiet
    :param x:
    :param delta:
    :return:
    """
    return 1.0 / (1.0 + torch.exp(-x / delta))


# Compute the density of states
def density_of_states_classic(energy, eigenvalues, delta=0.0005):
    dos = soft_threshold(torch.abs(eigenvalues - energy), delta)
    dos=torch.mean(dos)
    return dos



def get_dos(hamiltonian, steps=100, delta = 0.05, density_of_states=density_of_states_classic):
    """
    Computes the density of states in the classical way.
    It does not scale wale for big system we recommend using LSquant.

    :param hamiltonian:(Tensor)  hamiltonian matrix
    :param steps:
    :param delta: Energy interval width (you may adjust this)
    :return:
    """

    # Diagonalize the Hamiltonian to obtain eigenvalues
    eigenvalues = torch.linalg.eigvalsh(hamiltonian)

    # Define energy range for plotting
    energy_range = torch.linspace(min(eigenvalues), max(eigenvalues), steps)

    # Compute density of states for each energy in the range
    dos_values = [density_of_states(energy, eigenvalues, delta) for energy in energy_range]

    return [ energy_range, dos_values,]