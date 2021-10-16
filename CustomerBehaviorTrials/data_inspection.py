import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def make_csv_files():
    """
    This function converts the text files of the data into csv files
    :return: None; saves the csv files
    """

    allocations = pd.read_excel('../datasets/CER/allocations.xlsx')[['ID', 'Code']]
    # Code 1 corresponds to the residential users
    allocations = allocations[allocations['Code'] == 1]
    allocations = allocations['ID']

    for i in range(1, 7):
        print(f'Processing the {i}th file')
        file = pd.read_csv(f'../datasets/CER/File{i}.txt', names=['ID', 'Time', 'Energy'], delimiter=' ')
        file = pd.merge(file, allocations, how='inner', on='ID')
        print('  Writing the file...')
        file.to_csv(f'datasets/CustomerBehaviorTrials/File{i}.csv', index=False)


def group_maker():
    """
    This function processes the CBT dataset files and generates separate data files corresponding to
    each tariff group
    :return: None; saves the tariff group datasets.
    """
    allocations = pd.read_excel('../datasets/CER/allocations.xlsx')
    # Extract the residential consumers only
    allocations = allocations[allocations['Code'] == 1]
    # To make the dataset consistent we convert the 'b' as tariff allocation to 'B'
    allocations[allocations['Residential - Tariff allocation'] == 'b'] = 'B'

    # Finding the unique tariffs in the dataset
    unique_tariffs = allocations['Residential - Tariff allocation'].values
    unique_tariffs = np.unique(unique_tariffs)

    group_ids = []

    # Extracting the user ids of each tariff group
    for tariff in unique_tariffs:
        group = allocations[allocations['Residential - Tariff allocation'] == tariff]['ID']
        group_ids.append(group)

    for group_id, tariff in zip(group_ids, unique_tariffs):
        print(f'[*] Processing for the Tariff Group {tariff}...')
        grouped = []
        for i in range(1, 7):
            data_file = pd.read_csv(f'../datasets/CER/File{i}.csv')
            grouped_data = pd.merge(data_file, group_id, how='inner', on='ID')
            grouped.append(grouped_data)
        new_data = pd.concat(grouped, ignore_index=True)
        print(f'[*] Writing the data for tariff group {tariff}')
        new_data.to_csv(f'../datasets/CER/group_{tariff}.csv', index=False)


def cluster_analysis(group_id, n_clusters=10, show_plot=True):
    """
    This function receives a group id (or tariff group), and the number of clusters, and then clusters the
    users of the corresponding tariff group to the specified number of clusters using the K-Means algorithm
    :param group_id: the id of the tariff group ('A', 'B', 'C', 'D', 'E', 'W')
    :param n_clusters: number of clusters
    :param show_plot: if True, plots the clustering results; the default value is True
    :return: The complete list of users data and their corresponding cluster assignments
    """
    # Loading the data associated to the specified group
    print(f'[*] Loading the group {group_id} data...')
    group_data = pd.read_csv(f'../datasets/CER/group_{group_id}.csv')

    # Extracting the unique user ids
    unique_ids = np.unique(group_data['ID'].values)

    # Finding the common time span among all users in the group
    user_data = group_data[group_data['ID'] == unique_ids[0]]
    user_data = user_data.sort_values('Time')
    common_dates = user_data['Time']

    print('[*] Finding the common dates and times...')
    for i in range(1, len(unique_ids)):
        user_data = group_data[group_data['ID'] == unique_ids[i]]
        user_data = user_data.sort_values('Time')['Time']
        common_dates = pd.merge(common_dates, user_data, how='inner', on='Time')

    # Preparing the data for clustering and dimensionality reduction
    print('[*] Preparing the data for clustering...')
    instances = []
    clusterer = KMeans(n_clusters=n_clusters)
    dim_reduce = PCA(n_components=2)
    combined_users = []
    for i in range(len(unique_ids)):
        user_data_i = group_data[group_data['ID'] == unique_ids[i]]
        user_data_i = user_data_i.sort_values('Time')
        user_data_i = pd.merge(user_data_i, common_dates, how='inner', on='Time')
        combined_users.append(user_data_i)
        instances.append(user_data_i['Energy'].values)

    combined_users = pd.concat(combined_users, ignore_index=True)
    print('[*] Clustering the data...')
    clusterer.fit(np.array(instances))
    cluster_labels = clusterer.predict(np.array(instances))
    if show_plot:
        print('[*] Performing dimensionality reduction for visualization...')
        instances = dim_reduce.fit_transform(np.array(instances))
        colors = [cluster_labels[i] for i in range(len(instances))]
        x, y = instances[:, 0], instances[:, 1]
        c_dict = dict()
        viridis = cm.get_cmap('viridis', n_clusters)
        for i, color in enumerate(viridis.colors):
            c_dict[i] = color[:-1]
        fig, ax = plt.subplots()
        for g in np.unique(colors):
            ix = np.where(colors == g)
            ax.scatter(x[ix], y[ix], c=c_dict[g], label=g, s=50)
        ax.legend()
        plt.title(f'Clustering Analysis for the tariff group {group_id}')
        plt.show()

    # Returning the user ids and their corresponding cluster assignments
    print('[*] Returning the cluster assignments')
    cluster_assignments = dict()
    for i in range(n_clusters):
        cluster_assignments[i] = []

    for i, cluster in enumerate(cluster_labels):
        cluster_assignments[cluster].append(unique_ids[i])

    return cluster_assignments, combined_users

