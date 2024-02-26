"""
The simplest way to compare two Ham.
"""

import torch


def ham_difference(ham_target_, ham_pred_):
    """
    Computes the difference between ham
    :param ham_target_: (Tensor)
    :param ham_pred_:(Tensor)
    :return: (float)
    """
    dif = 0
    for i, ham_target in enumerate(ham_target_):
        ham_pred = ham_pred_[i]
        flattened1 = ham_target
        flattened2 = ham_pred
        dif += torch.sum(torch.abs(flattened1 - flattened2))

    return dif
