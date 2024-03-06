import numpy as np
from .site import Site


class SiteList:
    """
    Class for Handling groups of sites local proprieties of  local proprieties of a system.
    """

    def __init__(self, labels=None, coordinates=None, prop=None):
        """
        Args:
            labels (str):  string of the following structure : "Atom_Orbital_AnotherProp"
            coordinates (Tensor or List): xyz coordinates.
            prop (dict):  dictionary with the site proprieties.
        """
        self.site_list = []
        if (labels is not None) and (coordinates is not None) and (prop is not None) :
            for uid, (label, coord) in enumerate(zip(labels, coordinates)):
                self.site_list.append(Site(uid, label, coord, prop[uid]))
        elif (labels is not None) and (coordinates is not None):
            for uid, (label, coord) in enumerate(zip(labels, coordinates)):
                self.site_list.append(Site(uid, label, coord))

    def __str__(self):
        str_site_list = ""
        for site in self.site_list:
            str_site_list += str(site) + "\n"
        return str_site_list

    def coordinates(self):
        """

        Returns: coordinates as column vectors

        """

        return np.array([site.coord for site in self.site_list], dtype=float).T

    def add(self, site):
        self.site_list.append(site)

    def get_sites(self, index=None):
        if index is None:
            return self.site_list
        else:
            return [self.site_list[idx] for idx in index]

    def get_prop(self, index=None):
        prop = [site.prop for site in self.site_list]
        if index is None:
            return prop
        else:
            return prop[index]

    def get_uids(self, index=None):
        uids = np.array([site.uid for site in self.site_list], dtype=int)
        if index is None:
            return uids
        else:
            return uids[index]

        #    def get_sites_coord_local(self):
        # """
        # Computes the sites coordinates in local coordinates.
        # Returns: np.array of sites coordinates
        #
        # """

        #        site_xyz = []

        #        for site_key in self.site_list.keys():
        # coord_xyz = xyz_to_local(self.site_list[site_key].xyz, self.lattice_vectors)
        #            coord_xyz = self.site_list[site_key].xyz
        #            site_xyz.append(coord_xyz)
        #        return np.array(site_xyz)

        #    def local_to_xyz(local_vector, lattice_vectors):
        # """
        # Converts local coordinates to coordinates in real space (xyz).

        # Args:
        #    local_vector (numpy.ndarray): The local coordinates.
        #    lattice_vectors (numpy.ndarray): The lattice vectors.

        # Returns:
        #     numpy.ndarray: The corresponding coordinates in real space (xyz).
    # """
    # #        inverse_lattice = np.linalg.inv(lattice_vectors)
    #        xyz_vector = np.dot(local_vector, inverse_lattice)
    #        return xyz_vector

    #    def xyz_to_local(xyz_vector, lattice_vectors):
    # """
    # Converts coordinates from real space (xyz) to local coordinates.

    # Args:
    #     xyz_vector (numpy.ndarray): The coordinates in real space (xyz).
    #   lattice_vectors (numpy.ndarray): The lattice vectors.

    #   Returns:
    #    numpy.ndarray: The corresponding local coordinates.
    #       """
#        inverse_lattice = np.linalg.inv(lattice_vectors)
#        local_vector = np.dot(xyz_vector, inverse_lattice)
#        return local_vector
