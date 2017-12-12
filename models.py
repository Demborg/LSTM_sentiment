import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BaselineModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=256, hidden_size=100, num_layers=1)
        self.output_layer = nn.Linear(100, 4)
        self.h0 = Variable(torch.randn(1,1,100)) 

    def forward(self, sequence):
        output, hn = self.rnn(sequence, self.h0)
        predictions = self.output_layer(output)
        return predictions
        


