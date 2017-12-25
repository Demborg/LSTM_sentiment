import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import settings


class BaselineModel(nn.Module):

    def __init__(self, **kwargs):
        """
        Initialize new baseline model.
        :keyword argument: hidden_size: int, number of hitten units.
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]

        self.rnn = nn.RNN(input_size=256, hidden_size=self.hidden_size, num_layers=1)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        self.h0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, sequence):
        sequence = sequence.permute(1, 0, 2)
        output, hn = self.rnn(sequence, self.h0)
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "BaseLine_{}".format(self.hidden_size)


class PureGRU(nn.Module):

    def __init__(self, **kwargs):
        """
        Initialize new PureGRU model.
        :keyword arguments:
        hidden_size: int, number of hitten units.
        num_layers: int, Number of recurrent layers
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]
        self.num_layers = kwargs["num_layers"]

        self.gru = nn.GRU(input_size=256, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        self.h0 = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_size))

    def forward(self, sequence):
        sequence = sequence.permute(1, 0, 2)
        output, hn = self.gru(sequence, self.h0)
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "PureGRU_h{}_l{}".format(self.hidden_size, self.num_layers)


class SimpleLSTM(nn.Module):

    def __init__(self, **kwargs):
        """
        Initialize new SimpleLSTM model.
        :keyword argument: hidden_size: int, number of hitten units.
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]

        self.lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size, num_layers=1)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        self.h0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, sequence):
        sequence = sequence.permute(1, 0, 2)
        output, hn = self.lstm(sequence, (self.h0, self.c0))
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "SimpleLSTM_{}".format(self.hidden_size)


class EmbeddingLSTM(nn.Module):
    """ LSTM Model that learns its own character embeddings. Max index 256 hardcoded. """

    def __init__(self, **kwargs):
        """
        Initalize new EmbeddingLSTM model.
        :keyword hidden_size: number of hidden units.
        :keyword num_layers: number of LSTM layers.
        :keyword embedding_dim: Dimensionality of the embeddings.
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]
        self.num_layers = kwargs["num_layers"]
        self.embedding_dim = kwargs["embedding_dim"]

        self.char_embeddings = nn.Embedding(256, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)

        self.h0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, sequence):
        embeds = self.char_embeddings(sequence)
        output, hn = self.lstm(embeds.permute(1, 0, 2), (self.h0, self.c0))
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "EmbeddingLSTM_h{}_l{}".format(self.hidden_size, self.num_layers)

class ConvLSTM(nn.Module):

    def __init__(self, **kwargs):
        """
        Initalize new ConvLSTM model.
        :keyword hidden_size: number of hidden units.
        :keyword num_layers: number of LSTM layers.
        :keyword embedding_dim: Dimensionality of the embeddings.
        :keyword kernel_size: Size of the kernel we use
        :keyword kernel_dim : Dimension of the kernel
        :keyword batch_size : Size of the batch
        """
        super().__init__()
        self.num_layers = kwargs["num_layers"]
        self.hidden_size = kwargs["hidden_size"]
        self.embedding_dim = kwargs["embedding_dim"]
        self.kernel_size = kwargs["kernel_size"]
        self.kernel_nb = kwargs["kernel_nb"]
        self.batch_size = kwargs["batch_size"]
        self.dropout = kwargs["dropout"]
        self.char_embeddings = nn.Embedding(256, self.embedding_dim)
     
     #CNN
        self.conv = nn.Conv2d(1, self.kernel_nb, (self.kernel_size, 256))
        self.dropout = nn.Dropout(self.dropout)

     #LSTM
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        self.h0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, sequence):
        embeds = self.char_embeddings(sequence)
        #CNN
        cnn_x = self.conv(embeds.permute(1,0,2).unsqueeze(1))
        cnn_out = self.dropout(cnn_x)
        #LSTM
        lstm_x, hn = self.lstm(embeds.permute(1, 0, 2), (self.h0, self.c0))
        lstm_out = self.output_layer(lstm_x)

        #CNN_LSTM
        cnn_lstm_out = torch.cat((cnn_x, lstm_out), 0)
        cnn_lstm_out = torch.transpose(cnn_lstm_out, 0, 1)
        return cnn_lstm_out

    def get_name(self):
        return "ConvLSTM_h{}_l{}".format(self.hidden_size, self.num_layers)

