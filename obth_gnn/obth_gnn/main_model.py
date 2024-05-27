"""
This is the basic BN GNN class
"""

import torch.nn as nn
from .custom_block import MegNetBlock, HamHeadMeg


class HGnn(nn.Module):
    def __init__(self,
                 edge_shape,
                 node_shape,
                 u_shape,
                 embed_size=[32, 32, 32],
                 ham_graph_emb=[4, 4, 4],
                 ham_output_size=[2,2,1],
                 n_blocks=3):
        super(HGnn, self).__init__()

        # Pre-process embedding
        self.embedding = MegNetBlock(edge_shape, node_shape, u_shape, embed_size=embed_size, inner_skip=True)

        # Core Model
        n_blocks = n_blocks
        self.core = nn.ModuleList()
        for i in range(n_blocks - 1):
            self.core.append(MegNetBlock(embed_size[0], embed_size[1], embed_size[2], embed_size=embed_size))

        # Hamiltonian Extraction 
        self.hamiltonian_head = HamHeadMeg(edge_shape=embed_size[0],
                                           node_shape=embed_size[1],
                                           u_shape=embed_size[2],
                                           embedded_graph_size=ham_graph_emb,
                                           output_size=ham_output_size)

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):
        """
        :param x: Node proprieties
        :param edge_index: edges as [[vi....][vj.....]]
        :param edge_attr:edge attributes
        :param state:the global state
        :param batch: the batch (group of the graph) from where the node comes
        :param bond_batch: the bach from where the dge comes from
        :return: updated_node values , updated_edges, updated_global_state
        """

        # Embedding
        if self.embedding is not None:
            x, edge_attr, state = self.embedding(x, edge_index, edge_attr, state, batch, bond_batch)

        # Update the graph
        for module in self.core:
            x, edge_attr, state = module(x, edge_index, edge_attr, state, batch, bond_batch)

        # Extract hamiltonian
        ham_ii, ham_ij, ij = self.hamiltonian_head(x, edge_index, edge_attr, state, batch, bond_batch)

        return ham_ii, ham_ij, ij
