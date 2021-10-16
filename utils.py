import torch

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from datetime import datetime, timedelta

pd.options.mode.chained_assignment = None


#####################################################
# Generalized functions. Appropriate for any dataset
def series_to_sequence(dataset: pd.DataFrame, target_column: str, input_lag: int, output_lag: int, remove_columns=None):
    """
    This function receives an instance-based pandas dataframe and converts it into a sequential numpy dataset
    :param dataset: a given dataset with pd.DataFrame format
    :param target_column: the column which constitutes the response
    :param input_lag: an integer determining the window size for the past data
    :param output_lag: an integer determining the window size for the future data
    :param remove_columns: a list of names determining the columns that must be removed beforehand; default is None
    :return: two numpy arrays, one indicating the features with the shape (number of instances, input lag, dimension)
    and the other indicating the labels (or responses) with the shape (number of instances, output lag)
    """

    # Removing the target column from the collection of columns
    x_columns = list(dataset.columns)
    # x_columns.remove(target_column)

    # Removing the redundant columns (if given any) from the collection of the columns
    if remove_columns is not None:
        for column in remove_columns:
            x_columns.remove(column)

    # Converting the remaining columns to a numpy array
    x = dataset[x_columns].values
    # Converting the target column tp a numpy array
    y = dataset[target_column].values

    # Acquiring the initial number of instances
    ds_length = dataset.shape[0]

    new_samples = []
    new_labels = []

    # Generating the sequential data
    # Note that if the initial number of instances is N and the input lag in m and the output lag is k,
    # the final number of instances will be (N-m-k)
    for i in range(0, ds_length - (input_lag + output_lag)):
        new_sample = x[i: i + input_lag]
        new_label = y[i + input_lag: i + input_lag + output_lag]

        new_samples.append(new_sample)
        new_labels.append(new_label)

    return np.array(new_samples, dtype=np.float), np.array(new_labels, np.float)


def train_val_test_separator(samples, labels, indices, train_ratio=0.9, val_ratio=0.1):
    """
    This function receives the samples and labels numpy arrays and separates the data into three segments of training,
    validation, and test. Note that the number of instances (the first element of the shape attribute) in samples and
    labels must match.
    :param samples: the instances corresponding to features
    :param labels: the instances corresponding to the labels or responses
    :param indices: a randomly ordered list of indices of instances for which the train, validation and test sets are
    extracted
    :param train_ratio: the percentage of the data used for the training
    :param val_ratio: the percentage of the training data used for validation
    :return: six numpy arrays: training samples, and training labels. validation samples and validation labels. test
    samples and test labels
    """

    ds_length = samples.shape[0]
    # indices = [i for i in range(ds_length)]
    # np.random.shuffle(indices)
    # Calculating the number of training instances
    train_val_length = int(train_ratio * ds_length)
    # Randomly choose the training samples from the dataset
    train_val_indices = indices[: train_val_length]
    # Set the remaining samples as the test data
    test_indices = indices[train_val_length:]

    # Randomly choose the validation samples from the training data
    val_length = int(val_ratio * len(train_val_indices))

    # Finalize the training, validation, and test samples and labels
    train_val_samples, train_val_labels = samples[train_val_indices], labels[train_val_indices]
    if val_length > 0:
        train_samples, train_labels = train_val_samples[:-val_length], train_val_labels[:-val_length]
        val_samples, val_labels = train_val_samples[-val_length:], train_val_labels[-val_length:]
    else:
        train_samples, train_labels = train_val_samples[:-1], train_val_labels[:-1]
        val_samples, val_labels = train_val_samples[-1], train_val_labels[-1]
    test_samples, test_labels = samples[test_indices], labels[test_indices]

    return train_samples, train_labels, val_samples, val_labels, test_samples, test_labels


def train_test_separator(samples, labels):
    """
    This function separates the dataset into train set and test set and simply ignores the validation set
    In this function, only the last sample is considered for testing and all the other samples are considered
    for training.
    :param samples: the instances corresponding to features.
    :param labels: the instances corresponding to the labels or responses
    :return: four numpy arrays: training samples, and training labels. test samples and test labels
    """

    train_samples = samples[:-1]
    test_sample = samples[-1].reshape(1, samples[-1].shape[0], samples[-1].shape[1])
    train_labels = labels[:-1]
    test_label = labels[-1].reshape(1, labels[-1].shape[0])

    return train_samples, train_labels, test_sample, test_label


def mean_absolute_percentage_error(true: torch.Tensor, predicted: torch.Tensor):
    """
    This function is an implementation of the Mean Absolute Percentage Error (MAPE)
    :param true: the ground truth values
    :param predicted: the predicted values
    :return: the MAPE corresponding to the provided values
    """
    error = torch.abs(true - predicted).mean() / true.abs().mean()

    return 30 * error


def mean_absolute_scaled_error(true, predicted, is_numpy=False):
    """
    This function implements the Mean Absolute Scaled Error (MASE)
    :param true: the ground truth values
    :param predicted: the predicted values
    :param is_numpy: handles the tensor and ndarray inputs separately
    :return:
    """
    if not is_numpy:
        errors = (true - predicted).abs()
        errors = errors.mean(dim=-1)
        denumerator = ((true[:, :-1] - true[:, 1:]).abs() + 0.5).mean(dim=-1)

        error = (errors / denumerator)

        return error.mean()
    else:
        errors = (true - predicted).abs()
        errors = errors.mean()
        denumerator = ((true[:-1] - true[1:]).abs() + 0.5).mean()

        error = (errors / denumerator)

        return error.mean()


#####################################################
# Functions specialized to London Smart Meter dataset
def cleaned_household_data(dataset, household_id, common_dates, weather_dataset=None):
    """
    This function receives the halfhourly_dataset pandas dataset and a household id and cleans the data
    :param dataset: a pandas dataset corresponding to the halfhourly dataset
    :param household_id: the id of a target household
    :param common_dates: the dates which is common for all datasets; data is processed for these dates
    :param weather_dataset: the dataset containing the weather information (optional)
    :return: several chunks of data
    """

    # Separate the part of data associated to the target household
    data = dataset[dataset['LCLid'] == household_id]
    data = pd.merge(data, common_dates, how='inner', on='tstp')

    if weather_dataset is not None:
        data = pd.merge(data, weather_dataset, how='inner', on='tstp')

    if 'Unnamed: 0' in list(data.columns):
        data = data.drop(columns=['Unnamed: 0'])
    if 'pressure' in list(data.columns):
        # because this column contain NaN values
        data = data.drop(columns=['pressure'])

    data_types = data.dtypes
    print('Dataset Length : ', len(data))
    # Handle the low anomalies (concatenates the rows that each of which correspond to an interval less than 30 minutes
    # to to a single row corresponding to a 30-minute interval)
    low_anomalies = get_low_anomalies(data)

    # Create a list to save different chunks of a dataset
    dataset_chunks = []

    # Get the numpy array of the dataset
    new_data = data.values

    # For each found low anomaly, concatenate the data and remove the anomaly rows
    for i in range(len(low_anomalies)):
        index = low_anomalies[i]['index']
        previous_date = low_anomalies[i]['previous_date']
        new_date = timedelta(minutes=30) + previous_date
        new_date = datetime.strftime(new_date, '%Y-%m-%d %H:%M:%S.%f')
        energy = low_anomalies[i]['energy']
        new_data[index] = np.array([household_id, new_date, energy])
        new_data = np.delete(new_data, index + 1, 0)

    # Generate a new cleaned data
    data = pd.DataFrame(new_data, columns=list(data.columns))

    # Handle the high anomalies (separate the rows that each of which correspond to an interval more than 30 minutes
    high_anomalies = get_high_anomalies(data)

    # Turn the date values of the data to one hot vector
    dummies = time_to_one_hot(data)

    # Update the energy values
    dummies['energy(kWh/hh)'] = data['energy(kWh/hh)'].astype(float)
    if weather_dataset is not None:
        must_add_columns = list(data.columns)[3:]

        for column in must_add_columns:
            dummies[column] = data[column].astype(data_types[column])

    if len(high_anomalies) > 0:
        for i in range(len(high_anomalies)):
            if i == 0:
                dataset_chunks.append(dummies[: high_anomalies[i]['index']])
            elif i == len(high_anomalies) - 1:
                dataset_chunks.append(dummies[high_anomalies[i - 1]['index']: high_anomalies[i]['index']])
                dataset_chunks.append(dummies[high_anomalies[i]['index']:])
            else:
                dataset_chunks.append(dummies[high_anomalies[i - 1]['index']: high_anomalies[i]['index']])
    else:
        dataset_chunks.append(dummies)

    return dataset_chunks


def get_household_complete_data(dataset, household, common_dates, input_lag, output_lag, weather_dataset=None):
    """
    This function receives a pandas dataset, and a household id and returns the cleaned data
    :param dataset: a pandas dataset corresponding to the halfhourly dataset
    :param household: the household id
    :param common_dates: the dates which is common for all datasets; data is processed for these dates
    :param input_lag: the input lag for making the sequential data
    :param output_lag: the output_lag for making the sequential data
    :param weather_dataset: the dataset containing the weather information (optional)
    :return: an np feature matrix x, and an np response array y
    """
    if weather_dataset is None:
        chunks = cleaned_household_data(dataset, household, common_dates)
    else:
        chunks = cleaned_household_data(dataset, household, common_dates, weather_dataset)

    x_values = []
    y_values = []

    for chunk in chunks:
        if chunk.shape[0] >= input_lag + output_lag:
            series = series_to_sequence(chunk, 'energy(kWh/hh)', input_lag, output_lag)

            x_values.append(series[0])
            y_values.append(series[1])

    x = x_values[0]
    y = y_values[0]

    for i in range(1, len(x_values)):
        if len(x_values[i].shape) == 3:
            x = np.concatenate((x, x_values[i]), axis=0)
            y = np.concatenate((y, y_values[i]), axis=0)

    return x, y


def time_to_one_hot(dataset):
    """
    This function receives a pandas dataset and turns the dates to one hot vectors
    :param dataset: a pandas dataset
    :return: a new pandas dataset in which the date columns are changed into dummy variables
    """
    new_dataset = dataset.values

    for i in range(new_dataset.shape[0]):
        new_dataset[i][1] = new_dataset[i][1].split(' ')[1][:5]

    new_dataset = pd.DataFrame(new_dataset[:, 1], columns=['tstp'])

    dummies = pd.get_dummies(new_dataset['tstp'], prefix='time')

    return dummies


def get_low_anomalies(data):
    """
    This function receives a pandas dataframe and finds the rows that have a temporal difference with their previous
    rows less than 30 minutes
    :param data: a pandas dataframe
    :return: a list of dictionaries representing the information of the low anomaly rows
    """
    low_anomalies = []

    previous_date = datetime.strptime(str(data.iloc[0]['tstp'])[:-1], '%Y-%m-%d %H:%M:%S.%f')

    for i in range(1, len(data)):
        current_date = datetime.strptime(str(data.iloc[i]['tstp'])[:-1], '%Y-%m-%d %H:%M:%S.%f')

        difference = (current_date - previous_date).total_seconds() / 60

        if difference != 30.:
            difference_slots = difference / 30
            if difference_slots < 1:
                low_anomalies.append({'index': i,
                                      'state': 'less',
                                      'previous_date': previous_date,
                                      'diff': difference_slots})

        previous_date = current_date

    for i in range(len(low_anomalies) - 1):
        if low_anomalies[i]['diff'] + low_anomalies[i + 1]['diff'] == 1:
            energy1 = 0
            energy2 = 0
            if data.iloc[low_anomalies[i]['index']]['energy(kWh/hh)'] != 'Null':
                energy1 = float(data.iloc[low_anomalies[i]['index']]['energy(kWh/hh)'])
            if data.iloc[low_anomalies[i+1]['index']]['energy(kWh/hh)'] != 'Null':
                energy2 = float(data.iloc[low_anomalies[i+1]['index']]['energy(kWh/hh)'])

            low_anomalies[i]['energy'] = energy1 + energy2
            del(low_anomalies[i + 1])
            i += 1

    return low_anomalies


def get_high_anomalies(data):
    """
    This function receives a pandas dataframe and finds the rows that have a temporal difference with their previous
    rows higher than 30 minutes
    :param data: a pandas dataframe
    :return: a list of dictionaries representing the information of the high anomaly rows
    """
    high_anomalies = []

    previous_date = datetime.strptime(str(data.iloc[0]['tstp'])[:-1], '%Y-%m-%d %H:%M:%S.%f')

    for i in range(1, len(data)):
        current_date = datetime.strptime(str(data.iloc[i]['tstp'])[:-1], '%Y-%m-%d %H:%M:%S.%f')

        difference = (current_date - previous_date).total_seconds() / 60

        if difference != 30.:
            difference_slots = difference / 30

            if difference_slots > 1:
                high_anomalies.append({'index': i,
                                       'state': 'more',
                                       'previous_date': previous_date,
                                       'diff': difference_slots})
        previous_date = current_date

    return high_anomalies


################################################
# Visualization tools
def line_chart(values, index, names, index_name, title):
    """
    This function utilizes the plotly tools to draw line charts. The lines have same index (x-axis)
    but different values (y-axis)
    :param values: the values for the dependent variables
    :param index: the values for the independent variable
    :param names: the names of the dependent variables
    :param index_name: the name of the independent variable
    :param title: the title of the plot
    :return: None; draws the plot
    """
    fig = go.Figure()

    for value, name in zip(values, names):
        fig.add_trace(go.Scatter(x=index, y=value, name=name, line=dict(width=4)))

    fig.update_layout(title={
        'text': title,
        'y': 0.98,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
        xaxis_title=index_name,
        yaxis_title='Value')

    fig.show()


def draw_graph(graph):
    """
    This function utilizes the plotly tools to draw graphs
    :param graph: a nx graph instance
    :return: None; draws the graph
    """
    pos = nx.spring_layout(graph)

    for node in graph.nodes:
        graph.nodes[node]['pos'] = pos[node]

    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = graph.nodes[edge[0]]['pos']
        x1, y1 = graph.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlOrRd',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: ' + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.update_layout(
        title={
            'text': 'Sample Constructed Graph',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        })
    fig.show()
