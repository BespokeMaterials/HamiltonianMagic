"""

"""
import json
import numpy as np


class Site:
    """
    Class for headlining local proprieties of  local proprieties of a system.
    """

    def __init__(self, uid=0, label="X", coord=np.zeros(3), prop=None):
        """
        Args:
            uid (int):
            label (str):  string of the following structure : "Atom_Orbital_AnotherProp"
            coord (Tensor or List): xyz coordinates.
            prop (dict):  dictionary with the site proprieties.
        """

        self.uid = uid
        self.label = label
        self.coord = coord
        self.prop = prop

    def __str__(self):
        return str(self.uid) + " " + self.label + " " + str(self.coord[0]) + " " + str(self.coord[1]) + " " + str(
            self.coord[2])

    def get_coord(self):
        return self.coord

    def set_coord(self, coord):
        self.coord = coord

    def to_dictionary(self):
        """

        Returns: site as a dictionary.

        """
        site = {
            "site_id": self.uid,
            "label": self.label,
            "xyz": self.coord,
            "prop": self.prop
        }

        return site

    def to_json(self):
        """

        Returns: Site as json string

        """
        site = self.to_dictionary()
        json_string = json.dumps(site, indent=4)
        return json_string
