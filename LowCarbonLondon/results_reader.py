import os

import numpy as np

from prettytable import PrettyTable


candidate_blocks = [0, 19, 28, 68, 73, 83, 92, 93]


def report_results(block_id: int, model: str, **kwargs):
    """
    This function receives a block id and a model name and prints out the average performance of the
    specified model over the specified block id
    :param block_id: the block id for which we want to see the average performance
    :param model: the model for which we want to see the average performance; model = gcrnn, lstm, ffnn, svr, arima
    :param kwargs: if the model is selected to lstm, an optional boolean argument 'weather' can be passed
    :return: None; prints out the average performance of the model over the block id in a table
    """

    assert model == 'gcrnn' or model == 'lstm' or model == 'ffnn' or model == 'svr' or model == 'arima'

    path = f'results/block{block_id}/{model}'
    if not os.path.exists(path):
        raise Exception(f'[!] The results of the {model} on block {block_id} does not exist!'
                        f'First run the model over the data to store the results.')

    if model == 'lstm':
        if 'weather' in kwargs:
            if kwargs['weather']:
                path = f'results/block{block_id}/{model}/weather/'

    available_nodes = os.listdir(path)

    if f'block{block_id}_summary.txt' in available_nodes:
        available_nodes.remove(f'block{block_id}_summary.txt')

    table = PrettyTable(['User', 'MSE', 'RMSE', 'MAE', 'MAPE', 'MASE'])

    avg_mse = 0
    avg_rmse = 0
    avg_mae = 0
    avg_mape = 0
    avg_mase = 0

    skip = 0

    for i, node in enumerate(available_nodes):
        with open(f'{path}/{node}', 'r') as handle:
            result = handle.readlines()[0].split(',')

        mse = float(result[0].split(':')[1].strip())
        rmse = float(result[1].split(':')[1].strip())
        mae = float(result[2].split(':')[1].strip())
        mase = float(result[4].split(':')[1].strip())

        mape = result[3].split(':')[1].strip()
        if mape == 'nan' or mape == 'inf':
            skip += 1
            continue
        mape = float(mape)

        table.add_row([f'User {i}', mse, rmse, mae, mape, mase])

        avg_mse += mse
        avg_rmse += rmse
        avg_mae += mae
        avg_mape += mape
        avg_mase += mase

    avg_mse /= (len(available_nodes) - skip)
    avg_rmse /= (len(available_nodes) - skip)
    avg_mae /= (len(available_nodes) - skip)
    avg_mape /= (len(available_nodes) - skip)
    avg_mase /= (len(available_nodes) - skip)

    avg_mse = np.round(avg_mse, 3)
    avg_rmse = np.round(avg_rmse, 3)
    avg_mae = np.round(avg_mae, 3)
    avg_mape = np.round(avg_mape, 3)
    avg_mase = np.round(avg_mase, 3)

    table.add_row(['Average', avg_mse, avg_rmse, avg_mae, avg_mape, avg_mase])

    print(table)
    with open(f'results/block{block_id}/{model}/block{block_id}_summary.txt', 'w') as handle:
        handle.write(str(table))
