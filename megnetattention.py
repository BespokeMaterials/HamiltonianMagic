"""
"""
import torch
import torch.nn as nn


class MEGNetAttention(nn.Module):
	def __init__(self, 
				edge_input_shape, 
				state_input_shape,
				)

	self.nodes_w=[]
	self.edges_w=[]
	self.global_w=[]

	def forward(self,):

		new_edges=self.update_edges(edges)

		new_nodes=self.update_nodes(edges)

		new_global=self.update_global()
