"""
Example Bn
"""
import os
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from hw.hw_model import HWizard
import matplotlib.pyplot as plt
from obth_gnn.reconstruct import basic_ham_reconstruction
from bn_structures_to_graph import MaterialDS, MaterialMesh, MyTensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

training_data = torch.load('BN_database/Graphs/aBN.pt')
train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True)

def create_directory_if_not_exists(directory_path):
    """
    Creates a directory if it does not already exist.

    Args:
    directory_path (str): The path of the directory to create.

    Returns:
    None
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def det_importance(targets):
    # Get unique values and their counts
    targets=targets.type(torch. int64) 
    values, counts = torch.unique(targets, return_counts=True)
    # Print the frequency of each value
    reciprocal_dict = {}
    for value, count in zip(values, counts):
        if int(value.item()) not in reciprocal_dict.keys():
            reciprocal_dict[int(value.item())] = count.item()
        else:
            reciprocal_dict[int(value.item())]  =reciprocal_dict[int(value.item())]+count.item()
        #print(f"{value.item()}: {count.item()} times")
    # print("Done")
    targets_scale = torch.tensor([1/reciprocal_dict[int(k)] for k in targets], requires_grad=False)
    return targets_scale

def hop_on_difference(pred, targets):
    ko = 100
    kh = 5
    #

    pred_0 = pred[0][:, 0]  # .reshape([pred[0].shape[0]*2])
    targets_0 = targets[0][:, 0]  # .reshape([pred[0].shape[0] * 2])
    pred_1 = pred[1][:, 0]  # .reshape([pred[1].shape[0] * 2])
    targets_1 = targets[1][:, 0]  # .reshape([pred[1].shape[0] * 2])
    #print("pred", pred)

    # Get importance
    #targets_scale_0 = det_importance(targets_0).to(DEVICE)
   # targets_scale_1 = det_importance(targets_1).to(DEVICE)
    #print(pred_0)
    d_onsite  =torch.abs(pred_0 - targets_0)
    d_onsite =d_onsite  #* targets_scale_0
    d_onsite = torch.sum(d_onsite)


    d_hop=torch.abs(pred_1 - targets_1)
    d_hop =d_hop #*targets_scale_1
    d_hop = torch.sum(d_hop)
    #print(f"onsite{d_onsite}-hop:{d_hop}")
    loss =    kh * d_hop+d_hop**2 + ko * d_onsite  #


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
    plt.savefig(path+name)
    plt.show()
    plt.clf()

def main ():

    # create the experimant archive
    exp_name="test_hw_model"
    create_directory_if_not_exists(exp_name)
    create_directory_if_not_exists(f"{exp_name}/rezults")
    create_directory_if_not_exists(f"{exp_name}/rezults/img")
    create_directory_if_not_exists(f"{exp_name}/rezults/txt")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    model = HWizard(edge_shape=51,
                    node_shape=2,
                    u_shape=2,
                    embed_size=[21, 10, 5],
                    ham_output_size=[2,2,1],
                    orbital_blocks=3,
                    pair_interaction_blocks=2,
                    onsite_depth=2,
                    ofsite_depth=2)
    model.to(device)

    training_data = torch.load('BN_database/Graphs/aBN_noxyz.pt', )
 
    train_dataloader = DataLoader(training_data, batch_size=3, shuffle=True, )
    val_loader = None

    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=200)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=500)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=2100)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=500)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=1000)

    # Now let´s see the results:
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    for ko, inputs in enumerate(train_dataloader):
        inputs.to(trainer.device)
        x = inputs.x.to(torch.float32)
        edge_index = inputs.edge_index.to(torch.int64).to(device)
        edge_attr = inputs.edge_attr.to(torch.float32).to(device)
        state = inputs.u.to(torch.float32).to(device)
        batch = inputs.batch.to(device)
        bond_batch = inputs.bond_batch.to(device)
        hii, hij, ij = model(x, edge_index, edge_attr, state, batch.to(trainer.device),
                             bond_batch.to(trainer.device))
        #print("hij:", hij)
        hii = hii.to("cpu")
        hij = hij.to("cpu")
        ij = ij.to("cpu")

        pred_mat_r = torch.zeros([len(hii), len(hii)])
        pred_mat_i = torch.zeros([len(hii), len(hii)])
        for i, hi in enumerate(hii):
            pred_mat_r[i][i] = hi[0]
            pred_mat_i[i][i] = hi[1]

        for i, hx in enumerate(hij):
            pred_mat_r[ij[0][i]][ij[1][i]] = hx[0]
            pred_mat_i[ij[0][i]][ij[1][i]] = hx[1]

        target_mat_r = torch.zeros([len(hii), len(hii)])
        target_mat_i = torch.zeros([len(hii), len(hii)])
        for i, hi in enumerate(inputs.onsite):
            target_mat_r[i][i] = hi[0]
            target_mat_i[i][i] = hi[1]
        for i, hx in enumerate(inputs.hop):
            target_mat_r[ij[0][i]][ij[1][i]] = hx[0]
            target_mat_i[ij[0][i]][ij[1][i]] = hx[1]

        dif_mat_i = target_mat_i - pred_mat_i

        dif_mat_r = target_mat_r - pred_mat_r

        path = f"{exp_name}/rezults"
        plot_matrx(target_mat_r, name=f'{path}/img/{ko}_tar_rmag.png', path=path)

        plot_matrx(pred_mat_r, name=f'{path}/img/{ko}_pred_rmag.png', path=path)

        plot_matrx(dif_mat_r, name=f'{path}/img/{ko}_dif_real.png', path=path)

        plot_matrx(dif_mat_i, name=f'{path}/img/{ko}_dif_imag.png', path=path)
        print("Done")
        print("maxx:",dif_mat_r.max() )
        print("min:", dif_mat_r.min())
        mat = dif_mat_r.detach().numpy()
        with open(f'{path}/txt/{ko}_dif_rea.txt', 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.3f')
        mat = target_mat_r.detach().numpy()
        with open(f'{path}/txt/{ko}_target_mat_r.txt', 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.3f')
        mat = pred_mat_r.detach().numpy()
        with open(f'{path}/txt/{ko}_pred_mat_r.txt', 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.3f')

        if ko == 3:
            break

if __name__ == "__main__":
    main()
