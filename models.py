import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
        self.h0 = Variable(torch.randn(1, 1, self.hidden_size))

    def forward(self, sequence):
        output, hn = self.rnn(sequence, self.h0)
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "BaseLine_{}".format(self.hidden_size)


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
        self.h0 = Variable(torch.randn(1, 1, self.hidden_size))
        self.c0 = Variable(torch.randn(1, 1, self.hidden_size))

    def forward(self, sequence):
        output, hn = self.lstm(sequence, (self.h0, self.c0))
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "SimpleLSTM_{}".format(self.hidden_size)

