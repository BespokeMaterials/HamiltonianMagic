from torch_geometric.data import Data


class MaterialGraph(Data):
    def __init__(self, x, edge_index, edge_attr, u):
        super(MaterialGraph, self).__init__()
        self.x = x  # Node features
        self.edge_index = edge_index  # Edge indices
        self.edge_attr = edge_attr  # Edge features
        self.u = u  # Global features
