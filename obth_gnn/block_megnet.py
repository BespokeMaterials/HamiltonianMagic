"""
Megnet block following
"Graph Networks as a Universal Machine Learning
Framework for Molecules and Crystals"
by Chi Chen, Weike Ye, Yunxing Zuo, Chen Zheng, and Shyue Ping Ong∗
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class ShiftedSoftplus(nn.Module):
    """
    Teh activation function
    """
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
        self.shift = nn.Parameter(torch.log(torch.tensor([2.])), requires_grad=False)

    def forward(self, x):
        return self.sp(x) - self.shift



class MegnetModule(MessagePassing):
    def __init__(self,
                 edge_shape,
                 node_shape,
                 u_shape,
                 inner_skip=False,
                 embed_size=[32, 32, 32],
                 ):
        super().__init__(aggr="mean")

        self.inner_skip = inner_skip

        # Edge update
        edge_update_input_size=embed_size[1]+embed_size[1]+embed_size[0]+embed_size[2]
        self.phi_e = nn.Sequential(
            nn.Linear( edge_update_input_size, 2*embed_size[0]),
            ShiftedSoftplus(),
            nn.Linear(2*embed_size[0], 2*embed_size[0]),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size[0], embed_size[0]),
            ShiftedSoftplus(),
        )

        # Node update
        node_update_input_size=embed_size[0]+embed_size[1]+embed_size[2]
        self.phi_v = nn.Sequential(
            nn.Linear(node_update_input_size, 2 * embed_size[1]),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size[1], 2 * embed_size[1]),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size[1], embed_size[1]),
            ShiftedSoftplus(),
        )

        # Global  update
        global_update_input_size=embed_size[1]+embed_size[0]+embed_size[2]
        self.phi_u = nn.Sequential(
            nn.Linear(global_update_input_size, 2 * embed_size[2]),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size[2], 2 * embed_size[2]),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size[2], embed_size[2]),
            ShiftedSoftplus(),
        )

        # Preprocessing passes
        self.preprocess_e = nn.Sequential(
            nn.Linear(edge_shape, 2 * embed_size[0]),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size[0], embed_size[0]),
            ShiftedSoftplus(),
        )

        self.preprocess_v = nn.Sequential(
            nn.Linear(node_shape, 2 * embed_size[1]),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size[1], embed_size[1]),
            ShiftedSoftplus(),
        )

        self.preprocess_u = nn.Sequential(
            nn.Linear(u_shape, 2 * embed_size[2]),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size[2], embed_size[2]),
            ShiftedSoftplus(),
        )


    def forward(self,x, edge_index, edge_attr, state, batch, bond_batch):
        """

        :param x:
        :param edge_index:
        :param edge_attr:
        :param state:
        :param batch:
        :param bond_batch:
        :return:
        """
        