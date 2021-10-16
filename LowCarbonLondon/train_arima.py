import torch
import os
import sys

import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from source.utils import mean_absolute_percentage_error, mean_absolute_scaled_error

#######################################################
# Reading and manipulating the data
block_id = 93
dataset_path = f'datasets/NewLondon/block{block_id}_npy_datasets/'
files = sorted(os.listdir(dataset_path))

tests_x = []
tests_y = []

for i in range(0, len(files), 6):
    tests_x.append(np.load(dataset_path + files[i]))
    tests_y.append(np.load(dataset_path + files[i + 1]))

print(f'[*] Dataset information:\n'
      f'Block {block_id},\n'
      f'Number of Nodes (Users): {len(tests_x)},\n'
      f'Number of Testing Instances: {tests_x[0].shape[0]}\n')

###################################
# Configuring necessary directories
os.makedirs('results', exist_ok=True)
os.makedirs(f'results/block{block_id}', exist_ok=True)
os.makedirs(f'results/block{block_id}/arima', exist_ok=True)

###################################
# Configuring the hyperparameters
num_nodes = len(tests_x)
seq_len = tests_x[0].shape[1]
input_dim = tests_x[0].shape[2]
output_dim = tests_y[0].shape[1]
target_node = int(input(f'[*] Specify the target node (from 0 to {num_nodes - 1}): '))

#################################
# Setting up the model, optimizer, loss function, and summary writer
print('[*] Setting up the model, optimizer, and the loss function...')

#################################
# Training, validating and testing the model
print('========================== ARIMA Testing Started ==========================')
data_x = tests_x[target_node]
data_y = tests_y[target_node]

avg_mse = 0.
avg_rmse = 0.
avg_mae = 0.
avg_mape = 0.
avg_mase = 0.

for i in range(data_x.shape[0]):
    sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
    sys.stdout.flush()
    sys.stdout.write(f'Processing the data {i + 1}/{len(data_x)}')
    sys.stdout.flush()
    y_tr = data_x[i, :, 0]
    model = ARIMA(endog=y_tr, order=(5, 1, 0), enforce_stationarity=False)
    results = model.fit(method_kwargs={"warn_convergence": False})
    y_pred = results.predict(1, 2)
    y_true = data_y[i]

    avg_mse += np.mean((y_pred - y_true) ** 2)
    avg_rmse += np.sqrt(np.mean((y_pred - y_true) ** 2))
    avg_mae += np.mean(np.abs(y_pred - y_true))
    avg_mape += mean_absolute_percentage_error(torch.tensor(y_true), torch.tensor(y_pred)).item()
    avg_mase += mean_absolute_scaled_error(torch.tensor(y_true), torch.tensor(y_pred), is_numpy=True).item()

avg_mse /= len(data_x)
avg_rmse /= len(data_x)
avg_mae /= len(data_x)
avg_mape /= len(data_x)
avg_mase /= len(data_x)

print()
print('Avg MSE: {:.3f}, Avg RMSE: {:.3f}, Avg MAE: {:.3f}, Avg MAPE: {:.3f}, Avg MASE: {:.3f}'
      .format(avg_mse, avg_rmse, avg_mae, avg_mape, avg_mase))
with open(f'results/block{block_id}/arima/block{block_id}_node{target_node}.txt', 'w') as handle:
    handle.write('Avg MSE: {:.3f}, Avg RMSE: {:.3f}, Avg MAE: {:.3f}, Avg MAPE: {:.3f}, Avg MASE: {:.3f}'
                 .format(avg_mse, avg_rmse, avg_mae, avg_mape, avg_mase))
print('========================== ARIMA Testing Ended ============================')
#################################
