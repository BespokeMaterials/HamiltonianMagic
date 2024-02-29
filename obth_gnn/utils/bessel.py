import torch as tr
import math


def bessel(r_d, rc, n):
    """
    Bessel basis function
    :param r_d: (float) module of the distance between atoms.
    :param rc: (float) maximum radius of neighbours.
    :param n: (int or float)the number of Bessel basis functions.
    :return:n_bessel
    """

    if r_d < 0:
        r_d = - 1 * r_d
    pi = tr.tensor(math.pi)
    fc_rij = fc(r_d)
    n_bessel = tr.sqrt(2 / rc) * (tr.sin(n * pi * r_d / rc) / (r_d)) * fc_rij

    return n_bessel


def fc(r_d):
    """
    Is the cosine cutoff function
    :param r_d:(float) module of the distance between atoms.
    :return:
    """
    pass