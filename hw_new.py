

from hw.hw_model import HWizard
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader
from dft_data_to_grphs import MaterialDS, MaterialMesh, MyTensor
import torch
from utils import save_spot, create_directory_if_not_exists
import os
torch.set_float32_matmul_precision('high')
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
    # print("ham_target_", ham_target_[0].shape,ham_target_[1].shape )
    # print("ham_pred_", ham_pred_[0].shape,  ham_pred_[0].shape)
    dif = 0


    for i, ham_target in enumerate(ham_target_):
        ham_pred = ham_pred_[i]
        flattened1 = ham_target.flatten()
        flattened2 = ham_pred.flatten()
        dif += torch.sum(torch.abs(flattened1 - flattened2))

    return dif
## End: Cost functions ##



def main(exp_name,train_data_path,test_data_path):
    create_directory_if_not_exists(f"EXPERIMENTS/{exp_name}")
    log_dir = f"EXPERIMENTS/{exp_name}/lightning_logs/"
    create_directory_if_not_exists(log_dir)

    logger = TensorBoardLogger(save_dir=log_dir, name=exp_name)
    model = HWizard(edge_shape=51,
                    node_shape=2,
                    u_shape=10,
                    embed_size=[20, 20, 10],
                    ham_output_size=[2,2,1],
                    orbital_blocks=2,
                    pair_interaction_blocks=2,
                    onsite_depth=2,
                    ofsite_depth=2)

    m_path= "/home/ICN2/atomut/HamiltonianMagic/EXPERIMENTS/HW_model_mixt4_spots_new/spot1/model_.pt"
    model.load_state_dict(torch.load(m_path))

    training_data = torch.load(train_data_path )[:2]
    test_data = torch.load(test_data_path )[:2]
    train_dataloader = DataLoader(test_data, batch_size=2, shuffle=True,num_workers=31 )
    test_dataloader = DataLoader(training_data, batch_size=1, shuffle=False,num_workers=31 )
    # Get the number of elements in each dataloader
    num_train_elements = len(train_dataloader.dataset)
    num_test_elements = len(test_dataloader.dataset)

    print(f"Number of elements in train_dataloader: {num_train_elements}")
    print(f"Number of elements in test_dataloader: {num_test_elements}")
    val_check_interval =0.9

    #Spot 1
    for _ in range(50):
        model.loss_function = ham_difference
        trainer = pl.Trainer(max_epochs = 10,
                             val_check_interval=val_check_interval,
                             logger=logger,
                             strategy='ddp_find_unused_parameters_true',
                             log_every_n_steps=3)

        trainer.fit(model, train_dataloader, test_dataloader)
        #model=trainer.model

        save_spot(model=model,
                  exp_name=exp_name,
                  spot_nr=1,
                  data=test_dataloader)

    print("# Spot 2")
    for _ in range(10):
        model.loss_function = hop_on_difference
        trainer = pl.Trainer(max_epochs=10,
                             val_check_interval=val_check_interval,
                             logger=logger,
                             strategy='ddp_find_unused_parameters_true',
                             log_every_n_steps=3
                             )
        trainer.fit(model, train_dataloader, test_dataloader)
    # model = trainer.model

    print("# Spot 2.0")
    model.loss_function = ham_difference
    trainer = pl.Trainer(max_epochs=500,
                         val_check_interval=val_check_interval,
                         logger=logger,
                         strategy='ddp_find_unused_parameters_true',
                         log_every_n_steps=3
                         )
    trainer.fit(model, train_dataloader, test_dataloader)

    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=2,
              data=test_dataloader)

    print("# Spot 3")
    model.loss_function = hop_on_difference
    trainer = pl.Trainer(max_epochs=200,
                         val_check_interval=val_check_interval,
                         logger=logger,
                         strategy='ddp_find_unused_parameters_true',
                         log_every_n_steps=3
                         )
    trainer.fit(model, train_dataloader, test_dataloader)
    # model = trainer.model

    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=3,
              data=test_dataloader)

    print("# Spot 4")
    model.loss_function = ham_difference
    trainer = pl.Trainer(max_epochs=2000,
                         val_check_interval=val_check_interval,
                         logger=logger,
                         strategy='ddp_find_unused_parameters_true',
                         log_every_n_steps=3
                         )
    trainer.fit(model, train_dataloader, None)
    # model = trainer.model

    save_spot(model=model,
              exp_name=exp_name,
              spot_nr=4,
              data=test_dataloader)



if __name__ == "__main__":
    exp_name = "HW_model_mixt4_spots_mini"
    train_data_path = "DATA/DFT/BN_DFT_GRAPH/test.pt"
    test_data_path = "DATA/DFT/BN_DFT_GRAPH/train.pt"
    main(exp_name, train_data_path, test_data_path)