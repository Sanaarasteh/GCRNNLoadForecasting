import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn

# defining the paths to the London datasets as a dictionary
dataset_paths = {
    'household_info': '../datasets/London/informations_households.csv',
    'acorn_groups': '../datasets/London/acorn_details.csv',
    'weather_daily': '../datasets/London/weather_daily_darksky.csv',
    'weather_hourly': '../datasets/London/weather_hourly_darksky.csv',
    'holidays': '../datasets/London/uk_bank_holidays.csv',
    'daily_block': '../datasets/London/daily_dataset/daily_dataset/',
    'hh_block': '../datasets/London/halfhourly_dataset/halfhourly_dataset/'
}

# These are the blocks which contain sufficient users with pairwise correlations
# higher than 0.5; we denote these blocks as candidate blocks
candidate_blocks = [0, 19, 28, 68, 73, 83, 92, 93]


def initial_statistics():
    """
    This function outputs simple statistics of the London dataset, including
    number of blocks, number of unique users in each block, and intersection of
    users in all blocks (repetitive users)
    :return: None; prints out the information mentioned above
    """

    # Question 1: How many blocks are available?
    # Note that the energy consumption data is stored at 'hh_block' file
    available_blocks = len(os.listdir(dataset_paths['hh_block']))
    print('[*] Number of available blocks : ', available_blocks)

    # Question 2: How many users does each block contain?
    unique_users = []
    unique_ids = []

    # Iterating through the available blocks and counting their unique users
    for i in range(available_blocks):
        sentence = f'[*] Analyzing block {i + 1}/{available_blocks}'
        sys.stdout.flush()
        sys.stdout.write('\b' * len(sentence))
        sys.stdout.flush()
        sys.stdout.write(sentence)
        sys.stdout.flush()
        # reading the csv file of the dataset and keeping only the 'LCLid' column
        # which contains the id of the consumers
        dataset = pd.read_csv(dataset_paths['hh_block'] + 'block_' + str(i) + '.csv')['LCLid']
        # finding the unique consumer ids
        users = np.unique(dataset.values)
        unique_users.append(len(users))
        unique_ids.append(users)

    print('[*] Number of unique users in each block : ')
    print(unique_users)

    # Question 3: Do some blocks have the same users in common?
    # checking each pair of blocks and searching for repeated users
    print('[*] Checking th common users:')
    for i in range(available_blocks):
        for j in range(i + 1, available_blocks):
            intersection = set(unique_ids[i]).intersection(set(unique_ids[j]))
            if len(intersection) > 0:
                print(f'[!] Intersection found between block {i} and {j}:', intersection)


def correlation_studies(block_id: int, plot_matrix: bool = True):
    """
    This function receives a block id, loads the data of the users of the block id
    and calculates the correlation matrix of the users.
    :param block_id: the target block id for which we want to compute the correlation matrix
    :param plot_matrix: if True, plots the correlation matrix; the default value is True
    :return: the correlation matrix of the users and plots the correlation matrix
    of the users, and the trimmed data frames.
    """

    print('[*] Correlation studies for Block ', block_id)
    dataset = pd.read_csv(dataset_paths['hh_block'] + f'block_{block_id}.csv')
    # removing the rows containing null values
    dataset = dataset[dataset['energy(kWh/hh)'] != 'Null']
    # converting the energy values to float type
    dataset['energy(kWh/hh)'] = dataset['energy(kWh/hh)'].astype(float)

    # finding unique users
    household_ids = list(np.unique(dataset['LCLid']))

    data_frames = []
    for household_id in household_ids:
        data_frames.append(dataset[dataset['LCLid'] == household_id])

    # trimming the users data to their common datetime intervals
    common_dates = data_frames[0]['tstp']
    for i in range(1, len(data_frames)):
        common_dates = pd.merge(common_dates, data_frames[i]['tstp'], how='inner', on='tstp')

    if len(common_dates) > 0:
        for i in range(len(data_frames)):
            data_frames[i] = pd.merge(data_frames[i], common_dates, how='inner', on='tstp')

        correlations = np.eye((len(household_ids)))
        # defining four sets of potential users each of which corresponds to a specific threshold
        potential_users1 = []
        potential_users2 = []
        potential_users3 = []
        potential_users4 = []
        for i in range(len(household_ids)):
            for j in range(i + 1, len(household_ids)):
                correlation = np.corrcoef(data_frames[i]['energy(kWh/hh)'].values,
                                          data_frames[j]['energy(kWh/hh)'].values)[0, 1]
                correlations[i, j] = correlation
                correlations[j, i] = correlation
                # defining four different correlation thresholds
                if correlation > 0.5:
                    potential_users1.append((i, j))
                if correlation > 0.6:
                    potential_users2.append((i, j))
                if correlation > 0.7:
                    potential_users3.append((i, j))
                if correlation > 0.8:
                    potential_users4.append((i, j))

        # plotting the correlation matrix
        if plot_matrix:
            os.makedirs('correlation_plots', exist_ok=True)

            plt.figure(figsize=(12, 10))
            sn.heatmap(pd.DataFrame(correlations))
            plt.title('# of users with corr > 0.5 :' + str(len(potential_users1)) + '|' +
                      '# of users with corr > 0.6 :' + str(len(potential_users2)) + '\n' +
                      '# of users with corr > 0.7 :' + str(len(potential_users3)) + '|' +
                      '# of users with corr > 0.8 :' + str(len(potential_users4)) + '\n')
            plt.savefig(f'correlation_plots/block{block_id}.png')

        return potential_users1, data_frames
    else:
        print('\t[*] No data is available')

        return None


def block_energy_plot(block_id: int, plot_values: bool = True):
    """
    This function receives a block id, finds the unique users, and plots the users energy
    consumptions vs time.
    :param block_id: The target block for which we want to plot the users load consumption
    :param plot_values: If True, plots the energy consumption values; the default value is True
    :return: the consumption patterns of the correlated users and plots the load consumptions
    and the correlated users.
    """

    print('[*] Energy plot for Block', block_id)

    potential_users, data_frames = correlation_studies(block_id, plot_matrix=False)

    # Extracting the users with pairwise correlations higher than 0.5
    target_users = set()
    for pair in potential_users:
        target_users.add(pair[0])
        target_users.add(pair[1])

    target_users = sorted(list(target_users))
    legends = [f'User {i}' for i in target_users]

    # Extracting the energy consumption data of desired users
    patterns = []
    for user in target_users:
        patterns.append(data_frames[user]['energy(kWh/hh)'].values)

    # Plotting the energy consumption of the users
    if plot_values:
        index = [i for i in range(len(data_frames[0]))]
        for pattern in patterns:
            plt.plot(index, pattern)

        plt.title(f'Energy Consumption Pattern for Target Users in the Block {block_id}')
        plt.legend(legends)
        plt.show()

    return patterns, target_users


def block_box_plots(block_id: int):
    """
    This function receives a block id and plots the boxplot of each unique user for
    comparison of consumption values distributions.
    :param block_id: The target block id for which we want to plot the box plots
    :return: None
    """

    print('[*] Box plot for the users of the Block', block_id)

    patterns, _ = block_energy_plot(block_id, plot_values=False)

    plt.boxplot([patterns[i] for i in range(len(patterns))], patch_artist=True)
    plt.title(f'Box Plot of the Target Users in Block {block_id}')
    plt.show()


def user_information(block_id: int):
    """
    This function receives a block id and prints out the ACORN groups of
    the unique users.
    :param block_id: The target block id for which we want to extract unique ACORN groups
    :return: None
    """
    print('[*] Obtaining target users information for Block', block_id)
    dataset = pd.read_csv(dataset_paths['hh_block'] + f'block_{block_id}.csv')
    dataset = dataset[dataset['energy(kWh/hh)'] != 'Null']
    dataset['energy(kWh/hh)'] = dataset['energy(kWh/hh)'].astype(float)

    household_ids = list(np.unique(dataset['LCLid']))

    _, target_users = block_energy_plot(block_id, plot_values=False)

    target_users = sorted(list(target_users))
    household_info = pd.read_csv(dataset_paths['household_info'])

    print('-----------------------------')
    print('|   user    |   Acorn Group  |')
    print('-----------------------------')
    for user in target_users:
        acorn_group = household_info[household_info['LCLid'] == household_ids[user]]['Acorn'].values[0]
        print(f'| {household_ids[user]} |    {acorn_group}     |')
        print('-----------------------------')
