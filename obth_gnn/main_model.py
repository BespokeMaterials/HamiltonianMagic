"""
This is the basic BN GNN class
"""

import torch
import torch.nn as nn
from custom_block import MegNetBlock, HamHeadMeg


class BNGNN(nn.Module):
    def __init__(self,
                 edge_shape,
                 node_shape,
                 u_shape,
                 embed_size=[32, 32, 32],
                 ham_graph_emb=[4, 4, 4]):
        super(BNGNN, self).__init__()

        # Pre-process embedding
        self.embed = MegNetBlock(edge_shape, node_shape, u_shape, embed_size=embed_size)

        # Core Model
        n_blocks = 3
        self.core = nn.ModuleList()
        for i in range(n_blocks - 1):
            self.blocks.append(MegNetBlock(edge_shape, node_shape, u_shape, embed_size=embed_size))

        # Hamiltonian Extraction 
        self.hamiltonian_head = HamHeadMeg(edge_shape, node_shape, u_shape, embedded_graph_size=ham_graph_emb)

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):

        if self.embeded is not None:
            x, edge_atr, state = self.embed(x, edge_index, edge_attr, state, batch, bond_batch)

        # Update the graph
        x, edge_atr, state = self.core(x, edge_index, edge_attr, state, batch, bond_batch)

        # Extract hamiltonian
        ham_ii, ham_ij, ij = self.self.hamiltonian_head(x, edge_index, edge_attr, state, batch, bond_batch)

        return ham_ii, ham_ij, ij
