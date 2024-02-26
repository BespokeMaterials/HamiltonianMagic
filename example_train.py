"""
Example
"""

import torch
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from obth_gnn import HGnn
from obth_gnn.reconstruct import basic_ham_reconstruction
from obth_gnn.cost_functions import ham_difference
from example_build_chain_database import *

training_data = torch.load('artificial_graph_database/line_nodes_10_color_1_2.pt')
train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)


class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0

            # Training loop
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                inputs.to(self.device)
                x = inputs.x
                edge_index = inputs.edge_index
                edge_attr = inputs.edge_attr
                state = inputs.u
                batch = inputs.batch
                bond_batch = inputs.bond_batch

                hii, hij, ij = self.model(x, edge_index, edge_attr, state, batch.to(self.device), bond_batch.to(self.device))
                h_out = basic_ham_reconstruction(hii, hij, ij,batch, self.device)
                loss = self.loss_fn(targets, h_out, )
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():
                if self.val_loader is not None:
                    for inputs, targets in self.val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        val_loss += self.loss_fn(outputs, targets).item()

                    # Print statistics
                    print(f"Epoch {epoch + 1}/{num_epochs}, "
                          f"Train Loss: {running_loss / len(self.train_loader):.4f}, "
                          f"Val Loss: {val_loss / len(self.val_loader):.4f}")
                else:
                    # Print statistics
                    print(f"Epoch {epoch + 1}/{num_epochs}, "
                          f"Train Loss: {running_loss / len(self.train_loader):.4f}, ")

def main ():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    model = HGnn(edge_shape=3,
                 node_shape=2,
                 u_shape=2,
                 embed_size=[32, 5, 7],
                 ham_graph_emb=[4, 4, 4])
    model.to(device)

    training_data = torch.load('artificial_graph_database/line_nodes_10_color_1_2.pt', )
    #TODO:Solve batch problem
    # At the moment it crushes for batch !=1 .
    train_dataloader = DataLoader(training_data, batch_size=3, shuffle=True, )
    val_loader = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)

    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device)

    trainer.train(num_epochs=100)


if __name__ == "__main__":
    main()
