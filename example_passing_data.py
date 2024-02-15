"""
Before going to the training let´s test a bit:
 - if the model is working
 - how data look like
 - a simple cost function that we will call direct cost function
    because it will directly compare the target and generated hamiltonian between them
"""

import torch
import networkx as nx
from torch.utils.data import DataLoader
from obth_gnn import HGnn
from obth_gnn.reconstruct import basic_ham_reconstruction
from obth_gnn.cost_functions import ham_difference
from example_build_chain_database import *

training_data = torch.load('artificial_graph_database/line_nodes_10_color_1_2.pt')
train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)
element = training_data[0]

# Let´s see an element of the dataset
graph = element[0]
ham = element[1]
print("graph:\n", graph)
print("ham:\n", ham)

g = torch_geometric.utils.to_networkx(graph, to_undirected=False)
nx.draw(g)
plt.show()

# Now let´s try to pass it tru a model to check that everything is ok .
# If this work we can go to the training

# Model
model = HGnn(edge_shape=3,
             node_shape=2,
             u_shape=2,
             embed_size=[32, 5, 7],
             ham_graph_emb=[4, 4, 4])

print("model: \n", model)

# Passing:
x = graph.x
edge_index = graph.edge_index
edge_attr = graph.edge_attr
state = graph.u.unsqueeze(0)
batch = MyTensor(np.zeros(x.shape[0])).long()
bond_batch = graph.bond_batch

# TODO: Look a bit at the info structure and
# For me the batch and bond batch are confusing. Please Let´s talk
print("Look a bit to the structure of the information that is passed to the model")
print("bond_batch", bond_batch)
print("start x :", x.shape)
print("start edge_index :", edge_index.shape)
print("start edge_attr :", edge_attr.shape)
print("state shape:", state.shape)
print("batch:", batch)

hii, hij, ij = model(x, edge_index, edge_attr, state, batch, bond_batch)
print("Output of the model:")
print(f" hii : {hii},\n hij :{hij},\n ij : {ij}")
print(f" hii shape: {hii.shape},\n hij shape:{hij.shape},\n ij shape: {ij.shape}")

r_ham = basic_ham_reconstruction(hii, hij, ij)
print("Reconstructed hamiltonian: \n", r_ham)
print("Reconstructed hamiltonian shape : \n", r_ham.shape)

# Now is time to explore the cost function
target_pred_difference = ham_difference(ham_target=ham, ham_pred=r_ham)
print("Difference between prediction and target ham:", target_pred_difference)
