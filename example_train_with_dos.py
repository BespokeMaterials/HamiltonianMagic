"""
Example train with DOS
TODO: the batch problem still needs to be solve
"""

import torch as tr
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from obth_gnn import HGnn
from obth_gnn.reconstruct import basic_ham_reconstruction
from obth_gnn.cost_functions.classical_denity_of_states import density_of_states_classic
from example_build_chain_database import *

training_data = torch.load('artificial_graph_database/line_nodes_10_color_1_2_dos.pt')
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)


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
                inputs, targets = inputs.to(self.device), targets

                inputs.to(self.device)
                x = inputs.x
                edge_index = inputs.edge_index
                edge_attr = inputs.edge_attr
                state = inputs.u
                batch = inputs.batch
                bond_batch = inputs.bond_batch

                self.optimizer.zero_grad()
                hii, hij, ij = self.model(x, edge_index, edge_attr, state, batch.to(self.device),
                                          bond_batch.to(self.device))
                h_out = basic_ham_reconstruction(hii, hij, ij, batch, self.device)
                y = [inputs.dos0, inputs.dos1]
                loss = self.loss_fn(y, h_out)

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
        return self.model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HGnn(edge_shape=3,
                 node_shape=2,
                 u_shape=2,
                 embed_size=[32, 5, 7],
                 ham_graph_emb=[4, 4, 4])
    model.to(device)

    training_data = torch.load('artificial_graph_database/line_nodes_10_color_1_2_dos.pt', )
    # TODO:Solve batch problem
    # At the moment it crushes for batch !=1 .
    train_dataloader = DataLoader(training_data, batch_size=3, shuffle=True, )
    val_loader = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    def dos_difference_classic(target_dos_, h_out_):
        t_dif = 0
        for i, h_out in enumerate(h_out_):
            target_dos = [target_dos_[0][i], target_dos_[1][i]]

            eigenvalues = torch.linalg.eigvalsh(h_out)
            print("eigv:" , eigenvalues[:10])
            output_dos = [density_of_states_classic(energy, eigenvalues) for energy in target_dos[0]]
            print("odos:", output_dos)
            # TODO: Make it more efficient thisn is horrible
            dif = [(x - y.to(device)) ** 2 for x, y in zip(output_dos, target_dos[1])]
            dif = sum(dif)
            t_dif += dif
        return t_dif

    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=dos_difference_classic,
                      optimizer=optimizer,
                      device=device)

    trainer.train(num_epochs=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    trainer.optimizer = optimizer


if __name__ == "__main__":
    main()
