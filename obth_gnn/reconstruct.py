"""
Functions to reconstruct the hamiltonian.
"""

import torch


def basic_ham_reconstruction(hii, hij, ij, device="cpu"):
    """
     Reconstructs the hamiltonian matrix from the:
     :param hii:(Tensor) diagonal terms of shape [nr_nodes, 1]
     :param hij: (Tensor) non-diagonal terms [nr_edges one, 1]
     :param ij: edge indexes
     :return: ham (Tensor)
     """

    # Put the diagonal terms
    ham = torch.diag(hii[:, 0])
    ham.to(device)
    nd = torch.zeros(hii.shape[0], hii.shape[0], device=device)
    nd[ij[0], ij[1]] = hij[:, 0]
    ham += nd
    return ham
