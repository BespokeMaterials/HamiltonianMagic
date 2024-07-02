"""
Example Bn
"""
from tb_tain_old import  save_spot
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from obth_gnn import HGnn
import matplotlib.pyplot as plt
from obth_gnn.reconstruct import basic_ham_reconstruction
from obth_gnn.cost_functions import ham_difference
from tb_data_to_graph import MaterialDS, MaterialMesh, MyTensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def det_importance(targets):
    # Get unique values and their counts
    targets = targets.type(torch.int64)
    values, counts = torch.unique(targets, return_counts=True)
    # Print the frequency of each value
    reciprocal_dict = {}
    for value, count in zip(values, counts):
        if int(value.item()) not in reciprocal_dict.keys():
            reciprocal_dict[int(value.item())] = count.item()
        else:
            reciprocal_dict[int(value.item())] = reciprocal_dict[int(value.item())] + count.item()
        # print(f"{value.item()}: {count.item()} times")
    # print("Done")
    targets_scale = torch.tensor([1 / reciprocal_dict[int(k)] for k in targets], requires_grad=False)
    return targets_scale


def hop_on_difference(pred, targets):
    ko = 100
    kh = 5
    #

    pred_0 = pred[0][:, 0]  # .reshape([pred[0].shape[0]*2])
    targets_0 = targets[0][:, 0]  # .reshape([pred[0].shape[0] * 2])
    pred_1 = pred[1][:, 0]  # .reshape([pred[1].shape[0] * 2])
    targets_1 = targets[1][:, 0]  # .reshape([pred[1].shape[0] * 2])
    # print("pred", pred)

    # Get importance
    targets_scale_0 = det_importance(targets_0).to(DEVICE)
    targets_scale_1 = det_importance(targets_1).to(DEVICE)
    # print(pred_0)
    d_onsite = torch.abs(pred_0 - targets_0)
    d_onsite = d_onsite * targets_scale_0
    d_onsite = torch.sum(d_onsite)

    d_hop = torch.abs(pred_1 - targets_1)
    d_hop = d_hop * targets_scale_1
    d_hop = torch.sum(d_hop)
    # print(f"onsite{d_onsite}-hop:{d_hop}")
    loss = kh * d_hop + d_hop ** 2 + ko * d_onsite  #

    # pred_0 = pred[0][:, 1]  # .reshape([pred[0].shape[0]*2])
    # targets_0 = targets[0][:, 1]  # .reshape([pred[0].shape[0] * 2])
    # pred_1 = pred[1][:, 1]  # .reshape([pred[1].shape[0] * 2])
    # targets_1 = targets[1][:, 1]  # .reshape([pred[1].shape[0] * 2])

    # Get importance
    # targets_scale_0 = det_importance(targets_0).to(DEVICE)
    # targets_scale_1 = det_importance(targets_1).to(DEVICE)

    # d_onsite_i = torch.abs(pred_0 - targets_0)
    # d_onsite_i = d_onsite_i * targets_scale_0
    # d_onsite_i = torch.sum(d_onsite_i)
    # d_hop_i = torch.abs(pred_1 - targets_1)
    # d_hop_i = d_hop_i * targets_scale_1
    # d_hop_i = torch.sum(d_hop_i)
    # print(f"onsite{d_onsite}-hop:{d_hop}")
    # loss_i = ko * d_onsite_i + kh * d_hop_i + d_hop_i ** 2
    loss = loss  # +loss_i

    return loss


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
            for inputs in self.train_loader:
                inputs = inputs.to(self.device)
                targets = (inputs.onsite, inputs.hop)

                self.optimizer.zero_grad()

                x = inputs.x.to(torch.float32)
                edge_index = inputs.edge_index.to(torch.int64)
                edge_attr = inputs.edge_attr.to(torch.float32)
                state = inputs.u.to(torch.float32)
                batch = inputs.batch
                bond_batch = inputs.bond_batch

                hii, hij, ij = self.model(x, edge_index, edge_attr, state, batch.to(self.device),
                                          bond_batch.to(self.device))

                pred = (hii, hij)

                loss = self.loss_fn(pred, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():
                if self.val_loader is not None and epoch % 5 == 0:
                    for inputs in self.val_loader:
                        inputs = inputs.to(self.device)
                        targets = (inputs.onsite, inputs.hop)

                        x = inputs.x.to(torch.float32)
                        edge_index = inputs.edge_index.to(torch.int64)
                        edge_attr = inputs.edge_attr.to(torch.float32)
                        state = inputs.u.to(torch.float32)
                        batch = inputs.batch
                        bond_batch = inputs.bond_batch

                        hii, hij, ij = self.model(x, edge_index, edge_attr, state, batch.to(self.device),
                                                  bond_batch.to(self.device))
                        pred = (hii, hij)

                        loss = self.loss_fn(pred, targets)

                        val_loss += loss.item()

                    # Print statistics
                    print(f"Epoch {epoch + 1}/{num_epochs}, "
                          f"Train Loss: {running_loss / len(self.train_loader):.4f}, "
                          f"Val Loss: {val_loss / len(self.val_loader):.4f}")

                else:
                    # Print statistics
                    print(f"Epoch {epoch + 1}/{num_epochs}, "
                          f"Train Loss: {running_loss / len(self.train_loader):.4f}, ")

        return self.model


def plot_matrx(matrix, name='heatmap.png', path=""):
    plt.show()
    matrix = matrix.detach().numpy()
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add color bar
    plt.title(name)
    plt.savefig(path + name)
    plt.show()
    plt.clf()


def main():


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    model = HGnn(edge_shape=51,
                 node_shape=2,  # 8
                 u_shape=10,
                 embed_size=[20, 10, 5],
                 ham_graph_emb=[5, 5, 5],
                 n_blocks=2)
    model.to(device)

    training_data = torch.load('/home/ICN2/atomut/HamiltonianMagic/DATA/BN_database/Graphs/aBN_noxyz.pt', )
    # TODO:Solve batch problem
    # At the moment it crushes for batch !=1 .
    exp_name="super"
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True, )
    test_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    val_loader = None

    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=200)
    torch.save(model.state_dict(), f"super_model_.pt")
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=0,
              data=test_dataloader)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=500)
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=1,
              data=test_dataloader)
    torch.save(model.state_dict(), f"super_model_.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=2100)
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=2,
              data=test_dataloader)
    torch.save(model.state_dict(), f"super_model_.pt")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=500)
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=3,
              data=test_dataloader)
    torch.save(model.state_dict(), f"super_model_.pt")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=1000)
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=4,
              data=test_dataloader)
    torch.save(model.state_dict(), f"super_model_.pt")



if __name__ == "__main__":
    main()
