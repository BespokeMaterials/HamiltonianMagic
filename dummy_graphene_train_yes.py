"""
Example

- https://github.com/QuantumLab-ZY/HamGNN/blob/main/HamGNN/models/layers.py

"""

import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from obth_gnn import HGnn

from dummy_graphene_construct_dummyh import *

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA is available!")
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # Get the name of the current GPU
    gpu_name = torch.cuda.get_device_name(0)  # Assuming you have at least one GPU
    print(f"Current GPU: {gpu_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    print("CUDA is not available. Using CPU.")


def plot_matrx(matrix, name='heatmap.png', path=""):
    matrix = matrix.detach().numpy()
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add color bar
    plt.title(name)
    plt.savefig(path+name)
    plt.show()

def det_importance(targets):
    # Get unique values and their counts
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

    # print("p1", pred[1].shape)
    #
    # print("to", targets[0].shape)
    # print("t1", targets[1].shape)

    pred_0 = pred[0][:, 0]  # .reshape([pred[0].shape[0]*2])
    targets_0 = targets[0][:, 0]  # .reshape([pred[0].shape[0] * 2])
    pred_1 = pred[1][:, 0]  # .reshape([pred[1].shape[0] * 2])
    targets_1 = targets[1][:, 0]  # .reshape([pred[1].shape[0] * 2])
    #print("pred", pred)

    # Get importance
    targets_scale_0 = det_importance(targets_0).to("cuda")
    targets_scale_1 = det_importance(targets_1).to("cuda")

    d_onsite  =torch.abs(pred_0 - targets_0)
    d_onsite =d_onsite  * targets_scale_0
    d_onsite = torch.sum(d_onsite)


    d_hop=torch.abs(pred_1 - targets_1)
    d_hop =d_hop*targets_scale_1
    d_hop = torch.sum(d_hop)
    #print(f"onsite{d_onsite}-hop:{d_hop}")
    loss =   ko * d_onsite + kh * d_hop+d_hop**2

    pred_0 = pred[0][:, 1]  # .reshape([pred[0].shape[0]*2])
    targets_0 = targets[0][:, 1]  # .reshape([pred[0].shape[0] * 2])
    pred_1 = pred[1][:, 1]  # .reshape([pred[1].shape[0] * 2])
    targets_1 = targets[1][:, 1]  # .reshape([pred[1].shape[0] * 2])

    # Get importance
    targets_scale_0 = det_importance(targets_0).to("cuda")
    targets_scale_1 = det_importance(targets_1).to("cuda")

    d_onsite_i = torch.abs(pred_0 - targets_0)
    d_onsite_i = d_onsite_i * targets_scale_0
    d_onsite_i = torch.sum(d_onsite_i)
    d_hop_i = torch.abs(pred_1 - targets_1)
    d_hop_i = d_hop_i * targets_scale_1
    d_hop_i = torch.sum(d_hop_i)
    # print(f"onsite{d_onsite}-hop:{d_hop}")
    loss_i = ko * d_onsite_i + kh * d_hop_i + d_hop_i ** 2
    loss = loss +loss_i

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
                if self.val_loader is not None and epoch%5==0:
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    model = HGnn(edge_shape=21,
                 node_shape=5,
                 u_shape=1,
                 embed_size=[42, 7, 10],
                 ham_graph_emb=[4, 4, 4],
                 n_blocks=7)

    model.to(device)
    # Access parameters and their types
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Parameter type: {param.dtype}")

    training_data = torch.load('artificial_graph_database/DummyGrapheneGraph/graphene_dm_00.pt', )
    valid_data = torch.load('artificial_graph_database/DummyGrapheneGraph/graphene_dm_02.pt', )
    # TODO:Solve batch problem
    # At the moment it crushes for batch !=1 .
    train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True, )
    val_loader = DataLoader(valid_data, batch_size=10, shuffle=True, )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)

    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)

    model = trainer.train(num_epochs=30)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True, )
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)

    model = trainer.train(num_epochs=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True, )
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)

    model = trainer.train(num_epochs=101)

    # Now let´s see the results:
    train_dataloader = DataLoader(valid_data, batch_size=1, shuffle=True)
    for ko ,inputs in enumerate(train_dataloader):
        inputs.to(trainer.device)
        x = inputs.x.to(torch.float32)
        edge_index = inputs.edge_index.to(torch.int64).to(device)
        edge_attr = inputs.edge_attr.to(torch.float32).to(device)
        state = inputs.u.to(torch.float32).to(device)
        batch = inputs.batch.to(device)
        bond_batch = inputs.bond_batch.to(device)
        hii, hij, ij = model(x, edge_index, edge_attr, state, batch.to(trainer.device),
                             bond_batch.to(trainer.device))
        print("hij:", hij)
        hii = hii.to("cpu")
        hij = hij.to("cpu")
        ij = ij.to("cpu")

        pred_mat_r = tr.zeros([len(hii), len(hii)])
        pred_mat_i = tr.zeros([len(hii), len(hii)])
        for i, hi in enumerate(hii):
            pred_mat_r[i][i] = hi[0]
            pred_mat_i[i][i] = hi[1]

        for i, hx in enumerate(hij):
            pred_mat_r[ij[0][i]][ij[1][i]] = hx[0]
            pred_mat_i[ij[0][i]][ij[1][i]] = hx[1]

        target_mat_r = tr.zeros([len(hii), len(hii)])
        target_mat_i = tr.zeros([len(hii), len(hii)])
        for i, hi in enumerate(inputs.onsite):
            target_mat_r[i][i] = hi[0]
            target_mat_i[i][i] = hi[1]
        for i, hx in enumerate(inputs.hop):
            target_mat_r[ij[0][i]][ij[1][i]] = hx[0]
            target_mat_i[ij[0][i]][ij[1][i]] = hx[1]

        dif_mat_i = target_mat_i - pred_mat_i

        dif_mat_r = target_mat_r - pred_mat_r

        path="img/train_img/"
        plot_matrx(target_mat_r, name=f'{ko}_tar_rmag.png', path=path)
        plot_matrx(pred_mat_r, name=f'{ko}_pred_rmag.png',path=path)

        plot_matrx(dif_mat_r, name=f'{ko}_dif_real.png', path=path)
        plot_matrx(dif_mat_i, name=f'{ko}_dif_imag.png', path=path)
        print("Done")

        mat = dif_mat_r.detach().numpy()
        with open(f'{ko}_dif_rea.txt', 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.3f')
        mat = target_mat_r.detach().numpy()
        with open(f'{ko}_target_mat_r.txt', 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.3f')
        mat = pred_mat_r.detach().numpy()
        with open(f'{ko}_pred_mat_r.txt', 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.3f')

        if ko==100:
            break



if __name__ == "__main__":
    main()
