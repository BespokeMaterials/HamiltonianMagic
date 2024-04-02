"""
MegNet block following
"Graph Networks as a Universal Machine Learning
Framework for Molecules and Crystals"
by Chi Chen, Weike Ye, Yunxing Zuo, Chen Zheng, and Shyue Ping Ong∗
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class ShiftedSoftPlus(nn.Module):
    """
    The activation function
    """

    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
        self.shift = nn.Parameter(torch.log(torch.tensor([7.])), requires_grad=True)

    def forward(self, x):
        return self.sp(x) - self.shift


class MegNetBlock(MessagePassing):
    def __init__(self,
                 edge_shape,
                 node_shape,
                 u_shape,
                 inner_skip=False,  # tels us if we read the original value or th evaluate after embedding
                 embed_size=[32, 32, 32],
                 ):
        super().__init__(aggr="mean")

        self.inner_skip = inner_skip

        # Edge update
        edge_update_input_size = embed_size[1] + embed_size[1] + embed_size[0] + embed_size[2]
        self.phi_e = nn.Sequential(
            nn.Linear(edge_update_input_size, 2 * embed_size[0]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embed_size[0], 2 * embed_size[0]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embed_size[0], 2 * embed_size[0]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embed_size[0], embed_size[0]),
            ShiftedSoftPlus(),
        )

        # Node update
        node_update_input_size = embed_size[0] + embed_size[1] + embed_size[2]
        self.phi_v = nn.Sequential(
            nn.Linear(node_update_input_size, 2 * embed_size[1]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embed_size[1], 2 * embed_size[1]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embed_size[1], embed_size[1]),
            ShiftedSoftPlus(),
        )

        # Global  update
        global_update_input_size = embed_size[1] + embed_size[0] + embed_size[2]
        self.phi_u = nn.Sequential(
            nn.Linear(global_update_input_size, 2 * embed_size[2]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embed_size[2], 2 * embed_size[2]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embed_size[2], embed_size[2]),
            ShiftedSoftPlus(),
        )

        # Preprocessing passes
        self.preprocess_e = nn.Sequential(
            nn.Linear(edge_shape, 2 * embed_size[0]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embed_size[0], embed_size[0]),
            ShiftedSoftPlus(),
        )

        self.preprocess_v = nn.Sequential(
            nn.Linear(node_shape, 2 * embed_size[1]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embed_size[1], embed_size[1]),
            ShiftedSoftPlus(),
        )

        self.preprocess_u = nn.Sequential(
            nn.Linear(u_shape, 2 * embed_size[2]),
            ShiftedSoftPlus(),
            nn.Linear(2 * embed_size[2], embed_size[2]),
            ShiftedSoftPlus(),
        )

    def embedding(self, x, edge_attr, state):
        """
        Upgrade x, edge_attr, state to the right dimension for the rest od the model.
        :param x: features of the x node
        :param edge_attr:edge_attr: ( torch. Tensor ) edge attributes
        :param state:( torch. Tensor ) global state
        :return:x, edge_attr, state
        """

        x = self.preprocess_v(x)
        edge_attr = self.preprocess_e(edge_attr)
        state = self.preprocess_u(state)

        return x, edge_attr, state

    def message(self, x_i, x_j, edge_attr):
        """
        # In this case the message is just the value of th edge_attr
        # it is called in propagate.
        # is hte message that starts from one node to another one

        :param x_i:( torch. Tensor ) features of the x_i node.
        :param x_j:( torch. Tensor ) features of the x_k node.
        :param edge_attr: ( torch. Tensor ) edge attributes
        :return:edge_attr updated edge attributes
        """

        return edge_attr

    def update(self, inputs, x, state, batch):
        """
        # function that updated the nodes

        :param inputs: the average of th edges.
        :param x: features of the x node
        :param state:( torch. Tensor ) global state
        :param batch:-TODO : clarify after discussion
        :return:
        """

        return self.phi_v(torch.cat((inputs, x, state[batch, :]), 1))

    def edge_update(self, x_i, x_j, edge_attr, state, bond_batch):
        """
        # function that updates the edges
        # it is used in edge updater

        :param x_i:( torch. Tensor ) features of the x_i node. it has the shape [nr_edges, node_feature.shape]
        :param x_j: ( torch. Tensor ) features of the x_j node. it has the shape [nr_edges, node_feature.shape]
        :param edge_attr: ( torch. Tensor ) edge attributes
        :param state: ( torch. Tensor ) global state
        :param bond_batch: -TODO : clarify after discussion
        :return:phi_e(x_i, x_j, edge_attr, state)
        """

        cat = torch.cat((x_i, x_j, edge_attr, state[bond_batch, :]), 1)

        return self.phi_e(cat)

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

        # Preprocessing 
        if self.inner_skip:
            # We "add" the value after the internal embedding
            x, edge_attr, state = self.embedding(x, edge_attr, state)
            x_skip = x
            edge_attr_skip = edge_attr
            state_skip = state
        else:
            # We "add" the original values the ones  before the internal embedding
            # This may rage shape problems
            x_skip = x
            edge_attr_skip = edge_attr
            state_skip = state
            x, edge_attr, state = self.embedding(x, edge_attr, state)

        if torch.numel(bond_batch) > 0:
            # if the ege batch is provided it updates the edges
            edge_attr = self.edge_updater(
                edge_index=edge_index, x=x, edge_attr=edge_attr, state=state, bond_batch=bond_batch
            )

        # Propagate the message and update the nodes
        # the function above will call message and update 
        x = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, state=state, batch=batch)
        # Extracting the mean values from the graph required to update the global state


        u_v = global_mean_pool(x, batch)

        u_e = global_mean_pool(edge_attr, bond_batch, batch.max().item() + 1)


        #Todod:here i will have th eproblem
        # Computing the new global ste of the graph
        state = self.phi_u(torch.cat((u_e, u_v, state), 1))

        # add the skipped value
        x += x_skip
        edge_attr += edge_attr_skip
        state += state_skip

        return x, edge_attr, state
