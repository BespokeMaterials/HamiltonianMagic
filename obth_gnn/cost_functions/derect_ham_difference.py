"""
The simplest way to compare two Ham.
"""

import torch


def ham_difference(ham_target, ham_pred):
    """
    Computes the difference between ham
    :param ham_target: (Tensor)
    :param ham_pred:(Tensor)
    :return: (float)
    """
    flattened1 = ham_target.view(-1)
    flattened2 = ham_pred.view(-1)
    dif=torch.linalg.norm(flattened1 - flattened2)
    return dif
