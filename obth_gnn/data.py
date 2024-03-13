from torch_geometric.data import Data


class MaterialGraph(Data):
    def __init__(self, x, edge_index, edge_attr, u,bond_batch,y=None, ham=None, dos0=None,dos1=None):
        super(MaterialGraph, self).__init__()
        self.x = x  # Node features
        self.edge_index = edge_index  # Edge indices
        self.edge_attr = edge_attr  # Edge features
        self.u = u  # Global features
        self.bond_batch=bond_batch #tels from withch betch is the edge
        self.y=y # target propriety
        self.ham=ham # target hamiltonian
        self.dos1=dos1
        self.dos0 = dos0
    def __cat_dim__(self, key, value, *args, **kwargs):
        """
        Ad extra dim when batched u.
        :param key:
        :param value:
        :param args:
        :param kwargs:
        :return:
        """
        if key == "u":
            return None
        if key == "y":
            return None
        if key == "ham":
            return None
        if key == "dos0":
            return None
        if key == "dos1":
            return None

        return super().__cat_dim__(key, value, *args, **kwargs)





