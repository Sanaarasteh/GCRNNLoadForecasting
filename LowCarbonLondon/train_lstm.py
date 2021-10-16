import torch
import os
import time

import numpy as np
import torch.optim as opt

from math import inf
from torch.utils.data import DataLoader

from source.models import SimpleLSTM
from source.dataloader import LSTMDatasetReader
from source.transforms import SimpleToTensor
from source.utils import mean_absolute_percentage_error, mean_absolute_scaled_error

#######################################################
# Reading and manipulating the data
block_id = 0
add_weather = True

if add_weather:
    dataset_path = f'../datasets/NewLondon/block{block_id}_npy_datasets_weather/'
else:
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
# batch_size = int(input('[*] Specify the batch size: '))
batch_size = 20
num_nodes = len(trains_x)
seq_len = trains_x[0].shape[1]
input_dim = trains_x[0].shape[2]
hidden_dim = 128
output_dim = trains_y[0].shape[1]
target_node = int(input(f'[*] Specify the target node (from 0 to {num_nodes - 1}): '))
epochs = 10

#################################
# Configuring necessary directories
print('[*] Setting up necessary directories for saving the model...')

os.makedirs('../checkpoints', exist_ok=True)
os.makedirs('checkpoints/lstm', exist_ok=True)
os.makedirs('../results', exist_ok=True)
os.makedirs(f'results/block{block_id}', exist_ok=True)
os.makedirs(f'results/block{block_id}/lstm', exist_ok=True)
if add_weather:
    os.makedirs('checkpoints/lstm/weather', exist_ok=True)
    os.makedirs(f'checkpoints/lstm/weather/block{block_id}', exist_ok=True)
    os.makedirs(f'checkpoints/lstm/weather/block{block_id}/node{target_node}', exist_ok=True)

    os.makedirs(f'results/block{block_id}/lstm/weather', exist_ok=True)
else:
    os.makedirs('checkpoints/lstm/no_weather', exist_ok=True)
    os.makedirs(f'checkpoints/lstm/no_weather/block{block_id}', exist_ok=True)
    os.makedirs(f'checkpoints/lstm/no_weather/block{block_id}/node{target_node}', exist_ok=True)

    os.makedirs(f'results/block{block_id}/lstm/no_weather', exist_ok=True)
#################################
# Setting up the datasets and data loaders
print('[*] Loading the data...')
train_dataset = LSTMDatasetReader(trains_x[target_node], trains_y[target_node], SimpleToTensor())
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

val_dataset = LSTMDatasetReader(vals_x[target_node], vals_y[target_node], SimpleToTensor())
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

test_dataset = LSTMDatasetReader(tests_x[target_node], tests_y[target_node], SimpleToTensor())
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

#################################
# Setting up the model, optimizer, loss function, and summary writer
print('[*] Setting up the model, optimizer, and the loss function...')
model = SimpleLSTM(input_dim, hidden_dim, output_dim)
optimizer = opt.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()

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
        x, y = sample['x'], sample['y']
        optimizer.zero_grad()

        h = torch.rand(1, x.size(0), hidden_dim)
        c = torch.rand(1, x.size(0), hidden_dim)

        model.hidden_state = h
        model.cell_state = c

        out = model(x).squeeze(0)

        loss = loss_fn(out, y)

        avg_train_loss += loss.item()
        avg_train_mape += mean_absolute_percentage_error(y, out).item()
        avg_train_mase += mean_absolute_scaled_error(y, out).item()

        loss.backward()
        optimizer.step()

    avg_train_loss /= len(train_loader)
    avg_train_mape /= len(train_loader)
    avg_train_mase /= len(train_loader)

    # Validating the model
    with torch.no_grad():
        avg_val_loss = 0.
        for i, sample in enumerate(val_loader):
            x, y = sample['x'], sample['y']

            h = torch.rand(1, x.size(0), hidden_dim)
            c = torch.rand(1, x.size(0), hidden_dim)

            model.hidden_state = h
            model.cell_state = c

            out = model(x).squeeze(0)

            loss = loss_fn(out, y)

            avg_val_loss += loss.item()

        avg_val_loss /= len(val_loader)

        avg_test_loss = 0.
        avg_test_rmse = 0.
        avg_test_mape = 0.
        avg_test_mae = 0.
        avg_test_mase = 0.

        # Testing the model
        for i, sample in enumerate(test_loader):
            x, y = sample['x'], sample['y']

            h = torch.rand(1, x.size(0), hidden_dim)
            c = torch.rand(1, x.size(0), hidden_dim)

            model.hidden_state = h
            model.cell_state = c

            out = model(x).squeeze(0)

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

    if avg_val_loss < best_val_loss:
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
        if add_weather:
            torch.save(model.state_dict(),
                       f'checkpoints/lstm/weather/block{block_id}/node{target_node}/london_model.pkl')

            with open(f'results/block{block_id}/lstm/weather/block{block_id}_node{target_node}.txt', 'w') as handle:
                handle.write('Avg MSE: {:.3f}, Avg RMSE: {:.3f}, Avg MAE: {:.3f}, Avg MAPE: {:.3f}, Avg MASE: {:.3f}'
                             .format(avg_test_loss, avg_test_rmse, avg_test_mae, avg_test_mape, avg_test_mase))
        else:
            torch.save(model.state_dict(),
                       f'checkpoints/lstm/no_weather/block{block_id}/node{target_node}/london_model.pkl')

            with open(f'results/block{block_id}/lstm/no_weather/block{block_id}_node{target_node}.txt', 'w') as handle:
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
