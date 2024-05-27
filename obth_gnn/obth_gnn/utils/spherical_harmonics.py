

import torch as tr
from  math import factorial,pi


pi = tr.tensor(pi)
def sph_harm_r(theta, phi, l,m):
    """
    Computes the real spherical harmoinic.
    :param theta:
    :param phi:
    :param l:
    :param m:
    :return:
    """


    #TODO: be carefull for the sign of E
    p_lm = legendre(l, m, tr.cos(theta))
    y_lm = tr.sqrt((2 * l + 1) / (4 * pi))* tr.cos(m * phi)*p_lm

    return y_lm


# Function to compute associated Legendre polynomial using recursion
def legendre(l, m, x):
    """
    Function to compute associated Legendre polynomial using recursion.

    :param l:
    :param m:
    :param x:
    :return:
    """
    if m == 0:
        return tr.sqrt((2 * l + 1) / (4 * pi)) * tr.sqrt(factorial(l - m) / factorial(l + m)) * tr.pow(x, m) * tr.special.eval_legendre(l, x)
    elif m > 0:
        return tr.sqrt((2 * l + 1) / (4 * pi)) * tr.sqrt(factorial(l - m) / factorial(l + m)) * tr.pow(x, m) * tr.special.eval_legendre(l, x) * tr.cos(m * tr.acos(x))
    else:
        return (-1) ** (-m) * legendre(l, -m, x)