import torch
import torch.nn as nn

from torch_geometric.nn import DenseGCNConv


class GCLSTMCell(nn.Module):
    """
    This class implements an LSTM cell which its gates are replaced by GCN layers (Eq. 9)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        :param input_dim: The dimension of the inputs
        :param hidden_dim: The dimension of hidden layer of GCNs
        :param output_dim: The dimension of the outputs
        """

        super(GCLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.gcn_ii = DenseGCNConv(in_channels=input_dim, out_channels=output_dim, bias=True)
        self.gcn_hi = DenseGCNConv(in_channels=hidden_dim, out_channels=output_dim, bias=True)
        self.gcn_if = DenseGCNConv(in_channels=input_dim, out_channels=output_dim, bias=True)
        self.gcn_hf = DenseGCNConv(in_channels=hidden_dim, out_channels=output_dim, bias=True)
        self.gcn_ig = DenseGCNConv(in_channels=input_dim, out_channels=output_dim, bias=True)
        self.gcn_hg = DenseGCNConv(in_channels=hidden_dim, out_channels=output_dim, bias=True)
        self.gcn_io = DenseGCNConv(in_channels=input_dim, out_channels=output_dim, bias=True)
        self.gcn_ho = DenseGCNConv(in_channels=hidden_dim, out_channels=output_dim, bias=True)

    def forward(self, x, adj, hidden_state, cell_state):
        """
        :param x: a bxNxd matrix representing the feature matrix of a graph
        :param adj: a bxNxN matrix representing the adjacency matrix of a graph
        :param cell_state: a bxNxh matrix representing the previous cell state
        :param hidden_state: a bxNxh matrix representing the previous hidden state
        :return: (hidden_state, cell_state); a tuple of new hidden state and cell state each of which is a bxNxh matrix
        """

        # assert if the next hidden dimension is different than the current hidden dimension
        assert cell_state.size(-1) == self.output_dim

        i = torch.sigmoid(self.gcn_ii(x, adj) + self.gcn_hi(hidden_state, adj))
        f = torch.sigmoid(self.gcn_if(x, adj) + self.gcn_hf(hidden_state, adj))
        g = torch.tanh(self.gcn_ig(x, adj) + self.gcn_hg(hidden_state, adj))
        o = torch.sigmoid(self.gcn_io(x, adj) + self.gcn_ho(hidden_state, adj))

        new_cell_state = f * cell_state + i * g
        new_hidden_state = o * torch.tanh(new_cell_state)

        return new_hidden_state, new_cell_state


class GCLSTM(nn.Module):
    """
    This class implements the whole GCRNN network using the GCRNN cells defined above
    """
    def __init__(self, input_dim, hidden_dim, seq_len):
        """
        :param input_dim: The dimension of the inputs
        :param hidden_dim: The dimension of the hidden layers of GCNs
        :param seq_len: The length of input sequences
        """

        super(GCLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.gclstm_cells = nn.ModuleList()

        # Unfolding the recurrent network
        for i in range(seq_len):
            self.gclstm_cells.append(GCLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim))

    def forward(self, x, adj, initial_hidden_state, initial_cell_state):
        """
        :param x: a sequence of feature matrices of the shape bxsxNxd
        :param adj: a sequence of adjacency matrices of the shape bxsxNxN
        :param initial_hidden_state: the initial hidden state of the shape bxNxh
        :param initial_cell_state: the initial cell state of the shape bxNxh
        :return: the last hidden_state and cell_state of the network
        """
        hidden_state = initial_hidden_state
        cell_state = initial_cell_state

        for i in range(len(self.gclstm_cells)):
            hidden_state, cell_state = self.gclstm_cells[i](x[:, i, :, :], adj[:, i, :, :], hidden_state, cell_state)

        return hidden_state, cell_state
