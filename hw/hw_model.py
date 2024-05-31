"""
This is the basic BN GNN class
"""

import torch.nn as nn
from .blocks import MegNetBlock, HamHeadMeg


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
        self.embedding =   0   #MegNetBlock(edge_shape, node_shape, u_shape, embed_size=embed_size, inner_skip=True)

        # OrbitalInteractions:
        #MegNetBlock(embed_size[0], embed_size[1], embed_size[2], embed_size=embed_size)
        n_blocks = n_blocks
        self.orbital_interaction = nn.ModuleList()
        for i in range(n_blocks - 1):
            self.core.append()

        # PairInteractions 
        self.pair_interaction = HamHeadMeg(edge_shape=embed_size[0],
                                           node_shape=embed_size[1],
                                           u_shape=embed_size[2],
                                           embedded_graph_size=ham_graph_emb,
                                           output_size=ham_output_size)

        # Onsite
        self.onsite=0

        # Ofsite 
        self.ofsite=0


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
            x0, edge_attr0, state0 = self.embedding(x, edge_index, edge_attr, state, batch, bond_batch)

        # Update the  orbital_interaction
        x1=x0
        edge_attr1=edge_attr0
        state1=state0
        for module in self.orbital_interaction:
            x1, edge_attr1, state1 = module(x1, edge_index, edge_attr1, state1, batch, bond_batch)

        # Update onsite
        x2=x1
        edge_attr2=edge_attr1
        state2=state1
        for module in self.onsite:
            x2, edge_attr2, state2 = module(x2, edge_index, edge_attr2, state2, batch, bond_batch)

        # Update ofsite 
        x3=x1 
        edge_attr3=edge_attr1
        state3=state1
        for module in self.pair_interaction:
            x3, edge_attr3, state3 = module(x3, edge_index, edge_attr3, state3, batch, bond_batch)

        for module in self.ofsite:
            x3, edge_attr3, state3 = module(x3, edge_index, edge_attr3, state3, batch, bond_batch)




        ij =edge_index
        ham_ij=edge_attr3
        ham_ii=x2
        return ham_ii, ham_ij, ij
