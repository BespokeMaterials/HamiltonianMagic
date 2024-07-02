"""
Experiment on old model
"""

import torch
import os
import numpy as np
from torch_geometric.loader import DataLoader
from obth_gnn import HGnn
from obth_gnn.cost_functions import ham_difference
from utils import generate_heatmap, create_directory_if_not_exists
# Required when w eimport data
from dft_data_to_grphs import MaterialDS, MaterialMesh, MyTensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


## Cost functions ##
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


def ham_difference(ham_target_, ham_pred_):
    """
    Computes the difference between ham
    :param ham_target_: (Tensor)
    :param ham_pred_:(Tensor)
    :return: (float)
    """

    dif = 0
    for i, ham_target in enumerate(ham_target_):
        ham_pred = ham_pred_[i]
        flattened1 = ham_target
        flattened2 = ham_pred
        dif += torch.sum(torch.abs(flattened1 - flattened2))

    return dif


## End: Cost functions ##


## Save spot  ##
def save_spot(exp_name, spot_nr, model, data):
    # Create directory
    create_directory_if_not_exists("EXPERIMENTS")
    create_directory_if_not_exists(f"EXPERIMENTS/{exp_name}")
    create_directory_if_not_exists(f"EXPERIMENTS/{exp_name}/spot{spot_nr}")
    path = f"EXPERIMENTS/{exp_name}/spot{spot_nr}"
    path_img = os.path.join(path, "img")
    create_directory_if_not_exists(path_img)
    path_txt = os.path.join(path, "txt")
    create_directory_if_not_exists(path_txt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Save model
    torch.save(model.state_dict(), f"{path}/model_.pt")

    # Save data
    for ko, inputs in enumerate(data):

        inputs = inputs.to(device)
        x = inputs.x.to(torch.float32)
        edge_index = inputs.edge_index.to(torch.int64)
        edge_attr = inputs.edge_attr.to(torch.float32)
        state = inputs.u.to(torch.float32)
        batch = inputs.batch
        bond_batch = inputs.bond_batch

        with torch.no_grad():
            hii, hij, ij = model(x, edge_index, edge_attr, state, batch, bond_batch)

        # Move tensors to CPU for further processing and numpy conversion
        hii = hii.cpu()
        hij = hij.cpu()
        ij = ij.cpu()

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



        target_mat_r = target_mat_r.detach().numpy()
        pred_mat_r = pred_mat_r.detach().numpy()
        dif_mat_r = dif_mat_r.detach().numpy()
        dif_mat_i = dif_mat_i.detach().numpy()
        pred_mat_i=pred_mat_i.detach().numpy()
        target_mat_i=target_mat_i.detach().numpy()
        generate_heatmap(target_mat_r, filename=f'{path_img}/{ko}_tar_hmat.png')
        generate_heatmap(pred_mat_r, filename=f'{path_img}/{ko}_pred_hmat.png')
        generate_heatmap(dif_mat_r, filename=f'{path_img}/{ko}_dif_hmat.png')

        generate_heatmap(dif_mat_i, filename=f'{path_img}/{ko}_dif_smat.png')
        generate_heatmap(pred_mat_i, filename=f'{path_img}/{ko}_pred_smat.png')
        generate_heatmap(target_mat_i, filename=f'{path_img}/{ko}_target_smat.png')

        print("Done")
        print("max:", dif_mat_r.max())
        print("min:", dif_mat_r.min())


        np.save(os.path.join(path_txt, f'{ko}_dif_mat_hmat.npy'), dif_mat_r)
        np.save(os.path.join(path_txt, f'{ko}_target_mat_hmat.npy'), target_mat_r)
        np.save(os.path.join(path_txt, f'{ko}_pred_mat_hmat.npy'), pred_mat_r)

        np.save(os.path.join(path_txt, f'{ko}_dif_mat_smat.npy'), dif_mat_i)
        np.save(os.path.join(path_txt, f'{ko}_target_mat_smat.npy'), target_mat_i)
        np.save(os.path.join(path_txt, f'{ko}_pred_mat_smat.npy'), pred_mat_i)

        print("Done")



# Example usage (you need to define your model and data):
# save_spot('experiment1', 1, your_model, your_data)

## End: Save spot ##


## TRAINER ##
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


## END: TRAINER ##


def main(exp_name, train_data_path, test_data_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    model = HGnn(edge_shape=51,
                 node_shape=2,  # 8
                 u_shape=10,
                 embed_size=[20, 10, 5],
                 ham_graph_emb=[5, 5, 5],
                 n_blocks=2)
    state_dict = torch.load("EXPERIMENTS/super/spot4/model_.pt")
    model.load_state_dict(state_dict)
    model.to(device)

    training_data = torch.load(train_data_path, )
    test_data = torch.load(test_data_path, )

    train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True, )
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, )
    val_loader = None

    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=0,
              data=test_dataloader)
    # SPOT A:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=1000)
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=1,
              data=test_dataloader)

    # SPOT B:
    print("SPOT B:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=500)
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=2,
              data=test_dataloader)

    # SPOT C:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=200)
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=3,
              data=test_dataloader)

    # SPOT D:
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
              spot_nr=4,
              data=test_dataloader)

    # # SPOT E:
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # trainer = Trainer(model,
    #                   train_loader=train_dataloader,
    #                   val_loader=val_loader,
    #                   loss_fn=hop_on_difference,
    #                   optimizer=optimizer,
    #                   device=device)
    # model = trainer.train(num_epochs=1000)
    # save_spot(model=model,
    #           exp_name=exp_name,
    #           spot_nr=5,
    #           data=test_dataloader)

    # SPOT F:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=10000)
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=6,
              data=test_dataloader)

    # SPOT E:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=hop_on_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=100)
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=7,
              data=test_dataloader)

    # SPOT F:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    trainer = Trainer(model,
                      train_loader=train_dataloader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device)
    model = trainer.train(num_epochs=100)
    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=8,
              data=test_dataloader)

if __name__ == "__main__":
    exp_name = "Classic_tb_NABN_16_001_ad"
    train_data_path = '/home/ICN2/atomut/HamiltonianMagic/DATA/BN_database/Graphs/aBN_noxyz.pt'#"DATA/TB/BN_TB_GRAPH/train.pt"
    test_data_path = '/home/ICN2/atomut/HamiltonianMagic/DATA/BN_database/Graphs/aBN_noxyz.pt'#"DATA/TB/BN_TB_GRAPH/test.pt"
    main(exp_name, train_data_path, test_data_path)
