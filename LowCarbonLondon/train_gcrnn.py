import torch
import os
import time

import numpy as np
import torch.optim as opt

from math import inf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from source.models import Network
from source.dataloader import LondonDatasetReader
from source.transforms import ToTensor
from source.utils import mean_absolute_percentage_error, mean_absolute_scaled_error

#######################################################
# Reading and manipulating the data
candidate_blocks = [0, 19, 28, 68, 73, 83, 92, 93]
block_id = candidate_blocks[7]
print(f'[*] Preparing the data associated to the Block {block_id}')

dataset_path = f'../datasets/NewLondon/block{block_id}_npy_datasets/'
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
batch_size = int(input('[*] Specify the batch size: '))
num_nodes = len(trains_x)
seq_len = trains_x[0].shape[1]
input_dim = trains_x[0].shape[2]
hidden_dim = int(input('[*] Specify the hidden dimension of GCRN network, e.g, 32, 64, 128, 256: '))
output_dim = trains_y[0].shape[1]
target_node = int(input(f'[*] Specify the target node (from 0 to {num_nodes - 1}): '))
epochs = int(input('[*] Specify the number of epochs: '))

#################################
# Configuring necessary directories
print('[*] Setting up necessary directories for saving the model...')
os.makedirs('tensorboard', exist_ok=True)
os.makedirs(f'tensorboard/block{block_id}', exist_ok=True)
os.makedirs(f'tensorboard/block{block_id}/node{target_node}', exist_ok=True)

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('checkpoints/gcrnn', exist_ok=True)
os.makedirs(f'checkpoints/gcrnn/block{block_id}', exist_ok=True)
os.makedirs(f'checkpoints/gcrnn/block{block_id}/node{target_node}', exist_ok=True)

os.makedirs('results', exist_ok=True)
os.makedirs(f'results/block{block_id}', exist_ok=True)
os.makedirs(f'results/block{block_id}/gcrnn', exist_ok=True)

#################################
# Setting up the datasets and data loaders
print('[*] Loading the data...')
train_dataset = LondonDatasetReader(trains_x, trains_y, target_node, ToTensor())
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

val_dataset = LondonDatasetReader(vals_x, vals_y, target_node, ToTensor())
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

test_dataset = LondonDatasetReader(tests_x, tests_y, target_node, ToTensor())
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

#################################
# Setting up the model, optimizer, loss function, and summary writer
print('[*] Setting up the model, optimizer, and the loss function...')
model = Network(input_dim, hidden_dim, seq_len, output_dim, target_node=target_node)
optimizer = opt.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()

# Initiating Tensorboard's summary writer to record training, validation, and testing loss curves
summary = SummaryWriter(log_dir=f'tensorboard/block{block_id}/node{target_node}/')

#################################
# Training, validating and testing the model
print('========================== Training Started ==========================')

best_val_loss = float(inf)
for e in range(epochs):
    start_time = time.time()
    avg_train_loss = 0.
    avg_train_mape = 0.
    avg_train_mase = 0.
    for i, sample in enumerate(train_loader):
        x, adj, y = sample['x'], sample['adj'], sample['y']
        optimizer.zero_grad()

        h = torch.rand(x.size(0), num_nodes, hidden_dim)
        c = torch.rand(x.size(0), num_nodes, hidden_dim)

        out = model(x, adj, h, c)

        loss = loss_fn(out, y)

        avg_train_loss += loss.item()
        avg_train_mape += mean_absolute_percentage_error(y, out).item()
        avg_train_mase += mean_absolute_scaled_error(y, out).item()

        loss.backward()
        optimizer.step()

    avg_train_loss /= len(train_loader)
    avg_train_mape /= len(train_loader)
    avg_train_mase /= len(train_loader)

    summary.add_scalar('Average Train Loss', avg_train_loss, e)
    summary.add_scalar('Average Train MAPE', avg_train_mape, e)
    summary.add_scalar('Average Train MASE', avg_train_mase, e)

    with torch.no_grad():
        # Validating the model
        avg_val_loss = 0.
        for i, sample in enumerate(val_loader):
            x, adj, y = sample['x'], sample['adj'], sample['y']

            h = torch.rand(x.size(0), num_nodes, hidden_dim)
            c = torch.rand(x.size(0), num_nodes, hidden_dim)

            out = model(x, adj, h, c)

            loss = loss_fn(out, y)

            avg_val_loss += loss.item()

        avg_val_loss /= len(val_loader)
        summary.add_scalar('Average Validation Loss', avg_val_loss, e)

        avg_test_loss = 0.
        avg_test_rmse = 0.
        avg_test_mape = 0.
        avg_test_mae = 0.
        avg_test_mase = 0.

        # Testing the model
        for i, sample in enumerate(test_loader):
            x, adj, y = sample['x'], sample['adj'], sample['y']

            h = torch.rand(x.size(0), num_nodes, hidden_dim)
            c = torch.rand(x.size(0), num_nodes, hidden_dim)

            out = model(x, adj, h, c)

            loss = loss_fn(out, y)

            mae = mae_loss(out, y)

            avg_test_loss += loss.item()
            avg_test_rmse += np.sqrt(loss.item())
            avg_test_mae += mae.item()
            avg_test_mape += mean_absolute_percentage_error(y, out).item()
            avg_test_mase += mean_absolute_scaled_error(y, out).item()

        avg_test_loss /= len(test_loader)
        avg_test_rmse /= len(test_loader)
        avg_test_mae /= len(test_loader)
        avg_test_mape /= len(test_loader)
        avg_test_mase /= len(test_loader)

        summary.add_scalar('Test Average MSE Loss', avg_test_loss, e)
        summary.add_scalar('Test Average RMSE Loss', avg_test_rmse, e)
        summary.add_scalar('Test Average MAE Loss', avg_test_mae, e)
        summary.add_scalar('Test Average MAPE Loss', avg_test_mape, e)
        summary.add_scalar('Test Average MASE Loss', avg_test_mase, e)

    if avg_val_loss < best_val_loss:
        # Saving the model
        print('[*] Epoch: {}, '
              'Avg Train Loss: {:.3f}, '
              'Avg Train MAPE: {:.3f}, '
              'Avg Train MASE: {:.3f}, '
              'Avg Val Loss: {:.3f}, \n'
              'Avg Test Loss: {:.3f}, '
              'Avg Test RMSE: {:.3f}, '
              'Avg Test MAE: {:.3f}, '
              'Avg Test MAPE: {:.3f}, '
              'Avg Test MASE: {:.3f},'
              'Elapsed Time: {:.1f} (Model Saved)\n'
              .format(e, avg_train_loss, avg_train_mape, avg_train_mase, avg_val_loss, avg_test_loss, avg_test_rmse,
                      avg_test_mae, avg_test_mape, avg_test_mase, time.time() - start_time))
        torch.save(model.state_dict(), f'checkpoints/gcrnn/block{block_id}/node{target_node}/london_model.pkl')

        with open(f'results/block{block_id}/gcrnn/block{block_id}_node{target_node}.txt', 'w') as handle:
            handle.write('Avg MSE: {:.3f}, Avg RMSE: {:.3f}, Avg MAE: {:.3f}, Avg MAPE: {:.3f}, Avg MASE: {:.3f}'
                         .format(avg_test_loss, avg_test_rmse, avg_test_mae, avg_test_mape, avg_test_mase))

        best_val_loss = avg_val_loss
    else:
        print('[*] Epoch: {}, '
              'Avg Train Loss: {:.3f}, '
              'Avg Train MAPE: {:.3f}, '
              'Avg Train MASE: {:.3f}, '
              'Avg Val Loss: {:.3f}, \n'
              'Avg Test Loss: {:.3f}, '
              'Avg Test RMSE: {:.3f}, '
              'Avg Test MAE: {:.3f}, '
              'Avg Test MAPE: {:.3f}, '
              'Avg Test MASE: {:.3f},'
              'Elapsed Time: {:.1f}\n'
              .format(e, avg_train_loss, avg_train_mape, avg_train_mase, avg_val_loss, avg_test_loss, avg_test_rmse,
                      avg_test_mae, avg_test_mape, avg_test_mase, time.time() - start_time))

print('========================== Training Ended ============================')
#################################
