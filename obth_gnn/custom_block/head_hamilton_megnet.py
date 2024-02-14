"""
Head to extract hamiltonian from the graph.
Thi is used MegNet plus some extra layers.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import Set2Set

from .block_megnet import MegNetBlock, ShiftedSoftPlus


class HamHeadMeg(nn.Module):
    def __int__(self,
                edge_shape,
                node_shape,
                u_shape,
                embedded_graph_size=[3, 3, 3]):
        super(HamHeadMeg, self).__init__()

        self.graph_level = MegNetBlock(edge_shape, node_shape, u_shape, inner_skip=False,
                                       embed_size=embedded_graph_size, )
        self.se = Set2Set(embedded_graph_size[0], 1)
        self.sv = Set2Set(embedded_graph_size[1], 1)

        # convert the vector to the proper onsite or hopping value
        self.filter_v = nn.Sequential(
            nn.Linear(embedded_graph_size[1], 2 * embedded_graph_size[1]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embedded_graph_size[1], embedded_graph_size[1]),
            ShiftedSoftPlus(),
            nn.Linear(embedded_graph_size[1], 1),
        )
        self.filter_e = nn.Sequential(
            nn.Linear(embedded_graph_size[0], 2 * embedded_graph_size[0]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embedded_graph_size[0], embedded_graph_size[0]),
            ShiftedSoftPlus(),
            nn.Linear(embedded_graph_size[0], 1),
        )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):
        x, edge_attr, state = self.graph_level(x, edge_index, edge_attr, state, batch, bond_batch)
        x = self.sv(x, batch)
        edge_attr = self.se(edge_attr, bond_batch)

        h_ii = self.filter_v(x)
        h_ij = self.filter_e(edge_attr)

        return h_ii, h_ij, edge_index
