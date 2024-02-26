"""
Functions to reconstruct the hamiltonian.
"""

import torch


def basic_ham_reconstruction(hii, hij, ij,batch, device="cpu"):
    """
     Reconstructs the hamiltonian matrix from the:
     :param hii:(Tensor) diagonal terms of shape [nr_nodes, 1]
     :param hij: (Tensor) non-diagonal terms [nr_edges one, 1]
     :param ij: edge indexes
     :param batch (Tensor): one dimensional tensor of integers it says from witch graph is etch node.
     :return: ham (Tensor):  dim [nr of hamiltonians, hamiltonian.shape]
     """

    #TODO:
    # This is not efficient from the point ov view of memory we should change it
    # Put the diagonal terms
    ham = torch.diag(hii[:, 0])
    ham.to(device)

    nd = torch.zeros(hii.shape[0], hii.shape[0], device=device)
    nd[ij[0], ij[1]] = hij[:, 0]
    ham += nd
    bloks=extract_diagonal_blocks(ham, batch)
    return bloks



def extract_diagonal_blocks(matrix, indices):
    blocks = []
    prev_index = None
    start_index = 0

    for i, index in enumerate(indices):
        if prev_index is None or prev_index != index:
            if prev_index is not None:
                end_index = i
                block = matrix[start_index:end_index, start_index:end_index]
                blocks.append(block)
                start_index = end_index
            prev_index = index

    # Add the last diagonal block
    if start_index < len(matrix):
        block = matrix[start_index:, start_index:]
        blocks.append(block)

    return blocks