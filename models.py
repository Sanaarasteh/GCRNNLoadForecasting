import torch

import torch.nn as nn

from source.layers import GCLSTM


class Network(nn.Module):
    """
    This class implements the GCRNN and a following MLP network for STLF
    """
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, target_node=0):
        """
        :param input_dim: The dimension of the inputs
        :param hidden_dim: The dimension of the hidden layer of GCN and MLP
        :param seq_len: The length of the input sequence
        :param output_dim: The dimension of the outputs
        :param target_node: The id of the target user
        """
        super(Network, self).__init__()

        self.target_node = target_node

        self.gcrn1 = GCLSTM(input_dim, hidden_dim, seq_len)

        self.lin1 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.lin2 = nn.Linear(int(hidden_dim / 2), output_dim)

    def forward(self, x, adj, initial_hidden_state, initial_cell_state):
        """
        :param x: a sequence of feature matrices of the shape bxsxNxd
        :param adj: a sequence of adjacency matrices of the shape bxsxNxN
        :param initial_hidden_state: the initial hidden state of the shape bxNxh
        :param initial_cell_state: the initial cell state of the shape bxNxh
        :return: The predictions for the target user and for the next 'output_dim' steps
        """
        assert self.target_node <= x.size(2)

        hidden_state, _ = self.gcrn1(x, adj, initial_hidden_state, initial_cell_state)

        # read_out = torch.mean(hidden_state, dim=1) + hidden_state[:, self.target_node, :]
        read_out = hidden_state[:, self.target_node, :]

        out = torch.relu(self.lin1(read_out))
        out = self.lin2(out)

        return out


class FFNN(nn.Module):
    """
    This class implements a simple three layer MLP
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        :param input_dim: The dimension of the inputs
        :param hidden_dim: The dimension of the hidden layer
        :param output_dim: The dimension of outputs
        """

        super(FFNN, self).__init__()

        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        :param x: The input vector of the shape bxs to the MLP
        :return: The prediction for the next 'output_dim' steps
        """
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = self.dense3(x)

        return x


class SimpleLSTM(nn.Module):
    """
    This class implements a simple one-layer LSTM network followed by an MLP
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        :param input_dim: The dimension of inputs
        :param hidden_dim: The dimension of the hidden layer
        :param output_dim: The dimension of outputs
        """

        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

        self.dense1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dense2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        self.hidden_state = None
        self.cell_state = None

    def forward(self, x):
        """
        :param x: The input sequence of the shape bxsxd to the LSTM network
        :return: The predictions for the next 'output_dim' step
        """
        x, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))

        x = torch.relu(self.dense1(self.hidden_state))
        x = self.dense2(x)

        return x

