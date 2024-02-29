import torch as tr


def compute_polar_azimuthal_angles(origin, vector):
    """
    Compute the polar and azimuthal angle.
    :param origin:oring vector portion ov atom 1
    :param vector: arrow vector position ov atom 1
    :return: polar and azimuthal angle theta and phi.
    """
    # Compute differences in coordinates
    dx = vector[0] - origin[0]
    dy = vector[1] - origin[1]
    dz = vector[2] - origin[2]

    # Compute polar angle
    r = tr.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    theta = tr.arccos((vector[2] - origin[2]) / r)

    # Compute azimuthal angle
    phi = tr.arctan2(dy, dx)

    return theta, phi
