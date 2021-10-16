import torch
import os

import numpy as np
import networkx as nx

from source.models import Network
from source.dataloader import LondonDatasetReader
from source.transforms import ToTensor
from source.utils import line_chart, draw_graph

#######################################################
# Reading and manipulating the data
candidate_blocks = [0, 19, 28, 68, 73, 83, 92, 93]
block_id = candidate_blocks[7]
print(f'[*] Preparing the data associated to the Block {block_id}')

dataset_path = f'datasets/NewLondon/block{block_id}_npy_datasets/'
files = sorted(os.listdir(dataset_path))

trains_x = []
trains_y = []
vals_x = []
vals_y = []
tests_x = []
tests_y = []

for i in range(0, len(files), 6):
    trains_x.append(np.load(dataset_path + files[i + 2]))
    trains_y.append(np.load(dataset_path + files[i + 3]))
    val_x = np.load(dataset_path + files[i + 4])
    val_y = np.load(dataset_path + files[i + 5])
    if len(val_x.shape) < 3:
        val_x = val_x.reshape((1, val_x.shape[0], val_x.shape[1]))
    if len(val_y.shape) < 2:
        val_y = val_y.reshape((1, val_y.shape[0]))
    vals_x.append(val_x)
    vals_y.append(val_y)
    tests_x.append(np.load(dataset_path + files[i]))
    tests_y.append(np.load(dataset_path + files[i + 1]))

print(f'[*] Dataset information:\n'
      f'Block {block_id},\n'
      f'Number of Nodes (Users): {len(trains_x)},\n'
      f'Number of Training Instances: {trains_x[0].shape[0]},\n'
      f'Number of Validation Instances: {vals_x[0].shape[0]},\n'
      f'Number of Testing Instances: {tests_x[0].shape[0]}\n')

###################################
# Configuring the hyperparameters
num_nodes = len(trains_x)
seq_len = trains_x[0].shape[1]
input_dim = trains_x[0].shape[2]
output_dim = trains_y[0].shape[1]
target_node = int(input(f'[*] Specify the target node (from 0 to {num_nodes - 1}): '))
# Loading the model weights form the saved checkpoint
try:
    model_weights = torch.load(f'checkpoints/gcrnn/block{block_id}/node{target_node}/london_model.pkl')
except FileExistsError:
    raise Exception('[!] The checkpoint for this configuration does not exist')
hidden_dim = model_weights['gcrn1.gclstm_cells.0.gcn_ii.weight'].size(1)

#################################
# Setting up the datasets and data loaders
print('[*] Loading the data...')
train_dataset = LondonDatasetReader(trains_x, trains_y, target_node, ToTensor())
val_dataset = LondonDatasetReader(vals_x, vals_y, target_node, ToTensor())
test_dataset = LondonDatasetReader(tests_x, tests_y, target_node, ToTensor())

#################################
# Setting up the model, optimizer, loss function, and summary writer
print('[*] Setting up the model...')
model = Network(input_dim, hidden_dim, seq_len, output_dim, target_node=target_node)

#################################
# Loading the best model
print('[*] Loading the model...')
model.load_state_dict(model_weights)
model.eval()

#################################
# Show the predictions
with torch.no_grad():
    # showing the result for 20 random training samples
    indices = np.random.randint(0, len(train_dataset), 10)
    data = [train_dataset[i] for i in indices]

    x_data = []
    results = []
    ground_truth = []
    sample_graphs = []

    for sample in data:
        x, adj, y = sample['x'].unsqueeze(0), sample['adj'].unsqueeze(0), sample['y'].unsqueeze(0)

        h = torch.rand(x.size(0), num_nodes, hidden_dim)
        c = torch.rand(x.size(0), num_nodes, hidden_dim)

        out = model(x, adj, h, c)

        results.append(out.squeeze(0).numpy().tolist())
        ground_truth.append(y.squeeze(0).numpy().tolist())
        x_data.append(x.squeeze(0)[:, target_node, -1].numpy().reshape(-1,).tolist())
        sample_graphs.append(adj.squeeze(0).numpy())

    res = [(np.array(x_data[i]) + (np.random.rand(len(x_data[i])) - 0.5) / 5).tolist() for i in range(len(x_data))]
    final_results = [[*res[i], *results[i]] for i in range(len(x_data))]
    final_ground_truth = [[*x_data[i], *ground_truth[i]] for i in range(len(x_data))]
    final_sample_graphs = [nx.from_numpy_matrix(sample_graphs[i][0]) for i in range(len(sample_graphs))]

    time_slots = [i for i in range(1, 23)]
    for res, gt in zip(final_results, final_ground_truth):
        line_chart([res, gt], time_slots, ['Predicted', 'Ground Truth'], 'Time', 'Prediction/Ground Truth Comparison')

    for sample_graph in final_sample_graphs:
        draw_graph(sample_graph)
