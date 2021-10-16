import os
import random

import numpy as np
import pandas as pd

from source.CustomerBehaviorTrials.data_inspection import cluster_analysis
from source.utils import series_to_sequence, train_val_test_separator


def prepare_data(group_id: str, n_users: int, input_lag: int, output_lag: int):
    """
    This function receives a group id (a tariff group), the desired number of users (for which we want to generate
    the graphs), the input lag (or the look back factor), and the output lag (or the look ahead factor). The function
    then clusters the users of the specified group, randomly selects 'n_users' users, generates the sequential data for
    the users and saves the data.
    :param group_id: the target tariff group ('A', 'B', 'C', 'D', 'E', 'W')
    :param n_users: the number of users we want ot extract from a cluster
    :param input_lag: the look back factor for making the sequential data
    :param output_lag: the look ahead factor for making the sequential data
    :return: None; saves the sequential data for the desired users of a tariff group
    """

    # Clustering the users of the specified tariff group
    clusters, users_data = cluster_analysis(group_id, show_plot=False)

    cluster_ids = list(clusters.keys())
    # sample 'n_users' users random from each cluster and prepare its data
    for cluster_id in cluster_ids:
        print(f'[*] Preparing the the data for the cluster {cluster_id}')
        if len(clusters[cluster_id]) > n_users:
            # sampling the users
            sampled_users = random.sample(clusters[cluster_id], n_users)

            for sampled_user in sampled_users:
                # extracting the data of the target user
                user_data = users_data[users_data['ID'] == sampled_user]

                # finding the temporal anomalies in the data an chunking the data
                data_chunks = find_anomalies(user_data)

                print(f'\\__[*] {len(data_chunks)} chunks found')
                new_data_x = []
                new_data_y = []
                print(f'  \\__[*] Processing the chunks')

                # turning each chunk of data into sequential format
                for data_chunk in data_chunks:
                    if len(data_chunk) >= input_lag + output_lag:
                        x, y = series_to_sequence(data_chunk, 'Energy', input_lag, output_lag, remove_columns=['ID'])
                        new_data_x.append(x)
                        new_data_y.append(y)

                # concatenating the prepared data
                data_x = np.concatenate(new_data_x, axis=0)
                data_y = np.concatenate(new_data_y, axis=0)

                # saving the data
                np.save(f'../datasets/CER/group_{group_id}_cluster_{cluster_id}_user_{sampled_user}_x.npy', data_x)
                np.save(f'../datasets/CER/group_{group_id}_cluster_{cluster_id}_user_{sampled_user}_y.npy', data_y)
        else:
            print(f'[!] Not enough users in cluster {cluster_id}')


def generate_train_val_test_data(group_id: str):
    """
    This function generates training, validation, and test datasets. The function looks for the generated data files
    by the prepare_data function.
    :param group_id: the target tariff group ('A', 'B', 'C', 'D', 'E', 'W')
    :return: None; save the training, validation, test data files of the selected users of tariff group 'group_id'
    """

    users_data_x = []
    users_data_y = []

    # finding the generated files by the 'prepare_data' function
    files = os.listdir('../datasets/CER/')
    files = list(filter(lambda x: f'group_{group_id}_cluster_' in x, files))
    files = sorted(files)

    # Loading the data files; for efficiency the function removes the generated files after loading
    for i, file in enumerate(files):
        print(f'({i + 1}/{len(files)})[*] Loading {file}...')
        if '_x.npy' in file:
            cluster_id = file.split('_')[3]
            user_id = file.split('_')[5]
            users_data_x.append({'cluster': cluster_id,
                                 'user': user_id,
                                 'data': np.load(f'../datasets/CER/{file}')})
            os.remove(f'../datasets/CER/{file}')
        elif '_y.npy' in file:
            cluster_id = file.split('_')[3]
            user_id = file.split('_')[5]
            users_data_y.append({'cluster': cluster_id,
                                 'user': user_id,
                                 'data': np.load(f'../datasets/CER/{file}')})
            os.remove(f'../datasets/CER/{file}')

    # Shuffling the data
    indices = [i for i in range(users_data_x[0]['data'].shape[0])]
    np.random.shuffle(indices)

    # preparing the train, validation, test datasets
    for user_data_x, user_data_y in zip(users_data_x, users_data_y):
        training_files = train_val_test_separator(user_data_x['data'], user_data_y['data'], indices)
        cluster_id = user_data_x['cluster']
        user_id = user_data_x['user']

        assert cluster_id == user_data_y['cluster']
        assert user_id == user_data_y['user']

        # creating necessary directories
        os.makedirs(f'../datasets/CER/group_{group_id}', exist_ok=True)
        os.makedirs(f'../datasets/CER/group_{group_id}/cluster_{cluster_id}', exist_ok=True)

        # saving the train, val, test files
        print(f'\\__[*] Saving data for cluster {cluster_id}, user {user_id}')
        np.save(f'../datasets/CER/group_{group_id}/cluster_{cluster_id}/user_{user_id}_train_x.npy', training_files[0])
        np.save(f'../datasets/CER/group_{group_id}/cluster_{cluster_id}/user_{user_id}_train_y.npy', training_files[1])
        np.save(f'../datasets/CER/group_{group_id}/cluster_{cluster_id}/user_{user_id}_val_x.npy', training_files[2])
        np.save(f'../datasets/CER/group_{group_id}/cluster_{cluster_id}/user_{user_id}_val_y.npy', training_files[3])
        np.save(f'../datasets/CER/group_{group_id}/cluster_{cluster_id}/user_{user_id}_test_x.npy', training_files[4])
        np.save(f'../datasets/CER/group_{group_id}/cluster_{cluster_id}/user_{user_id}_test_y.npy', training_files[5])


def find_anomalies(dataframe: pd.DataFrame):
    """

    This function receives a pandas dataframe and checks if there is any time difference other than half hour. If there
    are anomalies in the time differences, the function divides the data into several chunks of data.
    :param dataframe: the target pandas dataframe
    :return: a list of data chunks
    """

    time_span = dataframe['Time'].values
    time_differences = np.diff(time_span)
    unique_differences = np.unique(time_differences)
    unique_differences = unique_differences[unique_differences != 1]
    anomalies_locations = []

    for difference in unique_differences:
        locations = np.where(time_differences == difference)
        for location in locations[0]:
            anomalies_locations.append(location)

    anomalies_locations.sort()

    chunks = []
    prev_location = 0
    for location in anomalies_locations:
        data_chunk = dataframe[prev_location: location]
        data_chunk = time_handler(data_chunk)
        chunks.append(data_chunk)
        prev_location = location

    data_chunk = dataframe[prev_location:]
    data_chunk = time_handler(data_chunk)
    chunks.append(data_chunk)

    return chunks


def time_handler(dataframe):
    """
    This function receives a pandas dataframe and turns the time feature of the dataframe into one hot vector
    :param dataframe: the target pandas dataframe
    :return: a modified pandas dataframe in which the time feature has been turned into one-hot vectors
    """

    new_times = np.zeros((len(dataframe), 49))
    for i in range(len(dataframe)):
        time = int(str(dataframe.iloc[0]['Time'])[-4:-2])

        new_time = [0 for _ in range(49)]
        if time == 50:
            new_time[0] = 1
        else:
            new_time[time] = 1

        new_times[i] = new_time

    dataframe = dataframe.drop(columns=['Time'])
    for i in range(49):
        dataframe[f'T{i}'] = new_times[:, i]

    return dataframe


# prepare_data('D', 15, 20, 2)
# generate_train_val_test_data('D')
