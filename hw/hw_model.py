"""
This is the basic BN GNN class
"""


from .blocks import MeGNet, OrbitalInteractions, PairInteractions , Onsite, Ofsite
import pytorch_lightning as pl
import torch.nn as nn
import torch

class HWizard(pl.LightningModule):
    def __init__(self,
                 edge_shape,
                 node_shape,
                 u_shape,
                 embed_size=[32, 32, 32],
                 ham_output_size=[2,2,1],
                 orbital_blocks=2,
                 pair_interaction_blocks=1,
                 onsite_depth=2,
                 ofsite_depth=3):
        super(HWizard, self).__init__()

        self.loss_function=None

        # Pre-process embedding
        self.embedding = MeGNet(edge_shape, node_shape, u_shape, embed_size=embed_size, inner_skip=True)

        # OrbitalInteractions:
        #MegNetBlock(embed_size[0], embed_size[1], embed_size[2], embed_size=embed_size)
      
        self.orbital_interaction = nn.ModuleList()
        for i in range(orbital_blocks - 1):
            self.orbital_interaction.append(OrbitalInteractions(input_dim=embed_size, output_dim=embed_size))

        # PairInteractions 
        self.pair_interaction = nn.ModuleList()
        for i in range(pair_interaction_blocks):
            self.pair_interaction.append(PairInteractions(input_dim=embed_size, output_dim=embed_size))
        

        # Onsite
        self.onsite=Onsite(input_dim=embed_size, output_dim=ham_output_size, depth=onsite_depth)
        

        # Ofsite 
        self.ofsite=Ofsite(input_dim=embed_size, output_dim=ham_output_size, depth=ofsite_depth)
        


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
        x2, edge_attr2, state2 = module(x2, edge_index, edge_attr2, state2, batch, bond_batch)

        # Update ofsite 
        x3=x1 +x2
        edge_attr3=edge_attr1+edge_attr2
        state3=state1+state2

        
        for module in self.pair_interaction:
            x3, edge_attr3, state3 = module(x3, edge_index, edge_attr3, state3, batch, bond_batch)

        x3, edge_attr3, state3 = module(x3, edge_index, edge_attr3, state3, batch, bond_batch)




        ij =edge_index
        ham_ij=edge_attr3
        ham_ii=x2
        return ham_ii, ham_ij, ij


    # Lightning things: 
    def _common_ste(self, batch, batch_idx):
        

        # Almos tsure that this is not th eproper way: 
        inputs = batch
        targets = (inputs.onsite, inputs.hop)
        x = inputs.x.to(torch.float32)
        edge_index = inputs.edge_index.to(torch.int64)
        edge_attr = inputs.edge_attr.to(torch.float32)
        state = inputs.u.to(torch.float32)
        batch = inputs.batch
        bond_batch = inputs.bond_batch

    
        ham_ii, ham_ij, ij = self(x, edge_index, edge_attr, state, batch, bond_batch)
        pred = (ham_ii, ham_ij)
        loss = self.loss_function(pred, targets)  
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_ste(batch, batch_idx) 
        #print("loss train:",loss.item()) 
        self.log("loss/train", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_ste(batch, batch_idx) 
        #print("loss val:",loss.item())  
        self.log("loss/val", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer