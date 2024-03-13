"""
Head to extract hamiltonian from the graph.
Thi is used MegNet plus some extra layers.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import Set2Set

from .block_megnet import MegNetBlock, ShiftedSoftPlus


class HamHeadMeg(nn.Module):
    def __init__(self,
                 edge_shape,
                 node_shape,
                 u_shape,
                 embedded_graph_size,
                 output_size=[1, 1, 1]):
        super(HamHeadMeg, self).__init__()

        self.graph_level = MegNetBlock(edge_shape, node_shape, u_shape, inner_skip=True,
                                       embed_size=embedded_graph_size, )

        # convert the vector to the proper onsite or hopping value
        self.filter = MegNetBlock(embedded_graph_size[0], embedded_graph_size[1], embedded_graph_size[1],
                                  inner_skip=True,
                                  embed_size=output_size, )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):
        """
        Pas the input graph to a graph processing step.
        Pass the result to a filter step that will put the note vale and edges to a one-dimensional nr.
        :param x: Node proprieties
        :param edge_index: edges as [[vi....][vj.....]]
        :param edge_attr:edge attributes
        :param state:the global state
        :param batch: the batch (group of the graph) from where the node comes
        :param bond_batch: the bach from where the dge comes from
        :return: updated_node values , updated_edges, updated_global_state
        """

        x, edge_attr, state = self.graph_level(x, edge_index, edge_attr, state, batch, bond_batch)
        x, edge_attr, state = self.filter(x, edge_index, edge_attr, state, batch, bond_batch)

        h_ii = x
        h_ij = edge_attr

        return h_ii, h_ij, edge_index
