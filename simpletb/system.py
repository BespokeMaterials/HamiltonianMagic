import numpy as np
from .site import Site


class System:

    def __init__(self, site_list=None, lattice_vectors=None, dimensions=(1, 1, 1)):
        """

        Args:
            site_list ():
            lattice_vectors ():
        """
        self.neighbours = None
        self.hopping_function = None
        self.onsite_function = None
        self.onsite_list = None
        self.hopping_list = None
        self.compute_neighbours = None

        self.site_list = site_list
        self.lattice_vectors = lattice_vectors
        self.dimensions = dimensions

    def get_site_coordinates(self):
        """

        Returns: site coordinates in fractional coordinates and as column vectors

        """

        return self.__cartesian_to_fractional(self.site_list.coordinates())

    def get_sites(self):
        return self.site_list.get_sites()

    def set_compute_neighbours(self, get_neighbours):
        """
        At one moment get neighbours will be incorporated in system until then we use his function to set it.
        Args:
            get_neighbours: A function that returns the neighbouring indices  [[uid,[nei_uid...]]...]

        """
        self.compute_neighbours = get_neighbours

    def set_onsite_function(self, onsite_function):
        """
        A function that takes as the argument an object of type Site and returns a real number.
        Args:
            onsite_function (Callable[[Site], float]): A function that accepts a Site object and returns a float.
        """
        # This is a simple check to see if the function can accept a Site instance and return a float.
        # More sophisticated or specific checks might be required depending on the use case.
        try:
            assert isinstance(onsite_function(self, Site()), float), "Function does not return a float"
        except AssertionError as e:
            raise TypeError \
                ("onsite_function must be a callable that accepts a Site instance and returns a float.") from e
        self.onsite_function = onsite_function
        return True

    def set_hopping_function(self, hopping_function):
        """
        A function that takes as the argument an object of two objects Site_i and Site_j and returns a complex number.
        Args:
            hopping_function (Callable[[Site,Site], complex]): A function that accepts
            two  Site objects and returns a complex.
        """
        # This is a simple check to see if the function can accept a Site instance and return a float.
        # More sophisticated or specific checks might be required depending on the use case.
        try:
            assert isinstance(hopping_function(self, Site(), Site()), complex), "Function does not return a float"
        except AssertionError as e:
            raise TypeError(
                "hopping_function must be a callable that accepts a Site instance and returns a float.") from e
        self.hopping_function = hopping_function
        return True

    def get_neighbours(self):
        """
        Returns: Returns the neighbours if they are computed if not it computes them an after that it returns them.
        """
        if self.neighbours is not None:
            return self.neighbours
        else:
            return self.compute_neighbours(self)

    def get_hopping(self):
        """

        Returns: Returns the hopping and computes them if they are not computed yet.

        """

        if self.hopping_list is not None:
            return self.hopping_list
        else:
            return self.compute_hopping()

    def get_onsite(self):
        """

        Returns: Returns the onsite and computes them if they are not computed yet.

        """

        if self.onsite_list is not None:
            return self.onsite_list
        else:
            return self.compute_onsite()

    def compute_hopping(self):
        """

        Computes the hopping for the entire system.
        Returns: [[(site1, site2), hopping_value]...]

        """
        hoppings = []
        neighbours = self.get_neighbours()
        print("neighbours:",neighbours)
        for ng in neighbours:
            sa_uid=ng[0]
            for sb_uid in ng[1]:
                if sa_uid!=sb_uid:
                    sa = self.site_list.get_sites([sa_uid])[0]
                    sb = self.site_list.get_sites([sb_uid])[0]
                    hopping_value = self.hopping_function(self, sa, sb)
                    hoppings.append([sa_uid, sb_uid, hopping_value])

        self.hopping_list = hoppings
        return self.hopping_list

    def compute_onsite(self):
        """
        Computes the onsite for etch site
        Returns: [[site, onsite],...]

        """
        onsite_list = []
        sites = self.site_list.get_sites()
        print(sites)
        for site_ in sites:
            onsite_list.append([site_.uid, self.onsite_function(self, site_)])
        self.onsite_list = onsite_list

        return self.onsite_list

    def get_hopping_onsite_matrix(self):
        """

        Puts the hopping and onsite on a matrix.
        Returns: matrix of hopping and onsite

        """

        onsite = self.get_onsite()
        hopping = self.get_hopping()

        mat = np.zeros([len(onsite), len(onsite)],dtype=complex)
        for i, os in enumerate(onsite):
            mat[os[0]][os[0]] = os[1]

        print("hopping", hopping)
        for i, hop in enumerate(hopping):
            mat[hop[0]][hop[1]] = hop[2]

        return mat

    def __cartesian_to_fractional(self, cart):
        """
        Convert a set of cartesian coordinates (provided as column vectors) into fractional coordinates
        Args:
            cart:  a set of cartesian coordinates (provided as column vectors)

        Returns: fractional coordinates

        """

        return np.linalg.inv(self.lattice_vectors) @ cart

    def __fractional_to_cartesian(self, frac):
        """
        convert a set of fractional coordinates (provided as column vectors) into cartesian coordinates
        Args:
            frac: a set of fractional coordinates (provided as column vectors)

        Returns: cartesian coordinates

        """

        return self.lattice_vectors @ frac
