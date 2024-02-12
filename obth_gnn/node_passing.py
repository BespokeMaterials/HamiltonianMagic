"""
This is the class that will handle passing the message between neighbouring nodes.
"""

import torch
import torch.nn as nn















class MomentEvolution(nn.Module):
    def __init__(self):
        pass
    def forward(self):

        # Update edges

        # Update Nodes

        # Update Global

        # Aggregate Nodes

        # Aggregate edges




class LinearAggregation(nn.Module):
    def __init__(self, input_dim,output_shape ):
        """
        This is the class that will handle passing the message between neighbouring nodes.
        """

        super().__init__()

        self.weights = torch.nn.Parameter(torch.randn(input_shape, output_shape))
        self.biases = torch.nn.Parameter(torch.randn(()))

    def forward(self, x, abj=None, attention=None):
        """
        This function will pas the nodes of the graph. If the abj is provided the message is passed
        only between the neighbours  if the attention is passed then the
        importance of etch node will be dictated by the attention.

        :param x: a tensor with the node values  of the shape [batch, nr_nodes, nr_node_prop]
        :type x:torch.Tensor
        :param abj: Abject matrix as a tensor of shape [batch, nr_nodes, nr_nodes]
        :type abj:torch.Tensor
        :param attention:Attention matrix as a tensor of shape [batch, nr_nodes, nr_nodes]
        :type attention:torch.Tensor
        :return: updated  of node value of shape [batch, nr_nodes, new_nr_node_prop]
        :rtype:torch.Tensor
        """

        agr=None
        if attention is not None and abj is not None:
            agr = torch.matmul(abj, attention)
        elif attention is not None:
            agr =attention
        elif abj is not None:
            agr =abj

        if agr is not None:
            x = torch.matmul(agr, x)

        x = torch.nn.functional.linear(x, self.weights, self.bias)

        return x


