import os

import numpy as np

from prettytable import PrettyTable


def report_results(group_id: str, model: str):
    """
    This function receives a block id and a model name and prints out the average performance of the
    specified model over the specified block id
    :param group_id: the tariff group for which we want to see the average performance
    :param model: the model for which we want to see the average performance; model = gcrnn, lstm, ffnn, svr, arima
    :return: None; prints out the average performance of the model over the block id in a table
    """

    assert model == 'gcrnn' or model == 'lstm' or model == 'ffnn' or model == 'svr' or model == 'arima'

    path = f'results/group_{group_id}'

    available_clusters = os.listdir(path)
    cluster_id = int(input(f'[*] Enter a cluster from the available clusters - {available_clusters} : '))

    path = f'{path}/cluster_{cluster_id}/{model}'

    if not os.path.exists(path):
        raise Exception(f'[!] The results of the {model} on group {group_id} - cluster {cluster_id} does not exist!'
                        f'First run the model over the data to store the results.')

    available_nodes = os.listdir(path)

    if f'summary.txt' in available_nodes:
        available_nodes.remove(f'summary.txt')

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
    with open(f'results/group_{group_id}/cluster_{cluster_id}/{model}/summary.txt', 'w') as handle:
        handle.write(str(table))
