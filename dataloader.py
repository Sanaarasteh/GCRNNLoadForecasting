import numpy as np
from torch.utils.data import Dataset
from sklearn.neighbors import kneighbors_graph


class LondonDatasetReader(Dataset):
    """
    This class implements a PyTorch data loader for the LCL dataset customized for GCRNN
    """
    def __init__(self, samples, labels, target_node, transforms=None):
        """
        :param samples: a list containing numpy arrays representing the consumers instances
        :param labels: a numpy array representing the responses
        :param target_node: The id of the target user
        :param transforms: A composition of transforms applied to the returning tensors
        """

        self.samples = samples
        self.labels = labels
        self.num_instances = samples[0].shape[0]
        self.num_nodes = len(samples)
        self.target_node = target_node
        self.transforms = transforms

        self.adj_matrices = []
        self.feature_matrices = []

        # Generating the feature matrices and adjacency matrices
        for i in range(self.num_instances):
            samples_list = []
            for j in range(self.num_nodes):
                samples_list.append(samples[j][i])

            adj, x = self.__build_graph(samples_list, 5)

            self.adj_matrices.append(adj)
            self.feature_matrices.append(x)

        self.adj_matrices = np.stack(self.adj_matrices)
        self.feature_matrices = np.stack(self.feature_matrices)

    def __len__(self):
        return self.num_instances

    def __getitem__(self, item):
        x = self.feature_matrices[item]
        adj = self.adj_matrices[item]
        label = self.labels[self.target_node][item]

        # Generating the output samples
        sample = {'x': x, 'adj': adj, 'y': label}

        # Applying the transforms to the sample
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __build_graph(self, samples_list, n_neighbors):
        """
        Building the similarity graphs using kNNs
        :param samples_list: The list of energy consumptions of users
        :param n_neighbors: Number of neighbors in the kNN algorithm
        :return: The feature matrix and the adjacency matrix of the users graph
        """

        sequence_length = samples_list[0].shape[0]
        dimension = samples_list[0].shape[1]

        adjacency_matrices = np.zeros((sequence_length, self.num_nodes, self.num_nodes))
        feature_matrices = np.zeros((sequence_length, self.num_nodes, dimension))

        for i in range(sequence_length):
            x = np.zeros((self.num_nodes, dimension))
            for j, sample in enumerate(samples_list):
                x[j, :] = sample[i]

            feature_matrices[i] = x

            # Constructing the kNN graph of the users
            adj = kneighbors_graph(x, n_neighbors).toarray()
            adjacency_matrices[i] = adj

        return adjacency_matrices, feature_matrices


class CERDatasetReader(Dataset):
    """
    This class implements the data loader of the CBT dataset customized for GCRNN
    """
    def __init__(self, samples, labels, target_node, transforms=None):
        """
        :param samples: a list containing numpy arrays representing the consumers instances
        :param labels: a numpy array representing the responses
        :param target_node: The id of the target user
        :param transforms: A composition of transforms applied to the returning tensors
        """

        self.samples = samples
        self.labels = labels
        self.num_instances = samples[0].shape[0]
        self.num_nodes = len(samples)
        self.target_node = target_node
        self.transforms = transforms

        self.adj_matrices = []
        self.feature_matrices = []

        # Generating the feature matrices and adjacency matrices
        for i in range(self.num_instances):
            samples_list = []
            for j in range(self.num_nodes):
                samples_list.append(samples[j][i])

            adj, x = self.__build_graph(samples_list, 7)

            self.adj_matrices.append(adj)
            self.feature_matrices.append(x)

        self.adj_matrices = np.stack(self.adj_matrices)
        self.feature_matrices = np.stack(self.feature_matrices)

    def __len__(self):
        return self.num_instances

    def __getitem__(self, item):
        x = self.feature_matrices[item]
        adj = self.adj_matrices[item]
        label = self.labels[self.target_node][item]

        sample = {'x': x, 'adj': adj, 'y': label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __build_graph(self, samples_list, n_neighbors):
        """
        Building the similarity graphs using kNNs
        :param samples_list: The list of energy consumptions of users
        :param n_neighbors: Number of neighbors in the kNN algorithm
        :return: The feature matrix and the adjacency matrix of the users graph
        """
        sequence_length = samples_list[0].shape[0]
        dimension = samples_list[0].shape[1]

        adjacency_matrices = np.zeros((sequence_length, self.num_nodes, self.num_nodes))
        feature_matrices = np.zeros((sequence_length, self.num_nodes, dimension))

        for i in range(sequence_length):
            x = np.zeros((self.num_nodes, dimension))
            for j, sample in enumerate(samples_list):
                x[j, :] = sample[i]

            feature_matrices[i] = x

            # Constructing the kNN graph of the users
            adj = kneighbors_graph(x, n_neighbors).toarray()
            adjacency_matrices[i] = adj

        return adjacency_matrices, feature_matrices


class SimpleDatasetReader(Dataset):
    """
    This class implements the data loader for the FFNN
    """
    def __init__(self, samples, labels, transforms=None):
        """
        :param samples: a numpy array representing the consumer's instances
        :param labels: a numpy array representing the responses
        :param transforms: A composition of transforms applied to the returning tensors
        """

        self.samples = samples[:, :, 0]
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        sample = {'x': self.samples[item], 'y': self.labels[item]}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class LSTMDatasetReader(Dataset):
    """
    This class implements the data loader for the SimpleLSTM model
    """
    def __init__(self, samples, labels, transforms=None):
        """
        :param samples: a numpy arrays representing the consumer's instances
        :param labels: a numpy array representing the responses
        :param transforms: A composition of transforms applied to the returning tensors
        """

        self.samples = samples
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        sample = {'x': self.samples[item], 'y': self.labels[item]}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
