import re

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

"""
Hello, and welcome to my LSTM experiment in PyTorch.
This is a space to try out the PyTorch utilities that we wish to use in the lab.
Since we'll be working with sequential data lets begin with defining our own data loader.
I will begin with a character-based LSTM on the goblet_book data.
"""


class TextFileDataset(Dataset):
    """ Dataset corresponding to the sentences of a text file.
    This dataset just stores everything in a big list.
    Note that we are free to load the data however we wish as long as we
    implement the __len__() and __getitem__() methods.
    This allows us to perform much smarter data loading with the
    Yelp dataset without blowing up our RAM, and the rest of our
    program does not have to care.
    """

    def __init__(self, path, indices=None):

        # Save the data to a list
        with open(path, "r") as file:
            self.data = file.readlines()
            self.data = [re.sub(r'[^a-zA-Z\ ]+', '', line).lower() for line in filter(lambda x: len(x) > 8, self.data)]

        # Save our index lookup table for the one-hot encodings
        if indices != None:
            self.indices = indices
        else:
            self.indices = {}

            # Nested for loob, ugly - but we can do whatever we want. If we use one-hot encodings in the real lab
            #  a better method should be applied.
            idx = 0
            for line in self.data:
                for c in line:
                    if c not in self.indices:
                        self.indices[c] = idx
                        idx += 1
            self.max_idx = idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """ Let's let the getitem method convert the items to one-hot
         encodings and return them in a torch tensor """
        line = self.data[item]
        indicies = [self.indices[c] for c in line]
        line_array = np.zeros([len(indicies), self.max_idx], dtype="float32")

        for idx, line in enumerate(line_array):
            line_array[idx, indicies[idx]] = 1
        return torch.from_numpy(line_array)  # This conversion is super fast due to shared memory

# So now let's create a dataset now when our class is working
dataset = TextFileDataset("goblet_book.txt")

# This dataset can be intexed just like any collection
print(dataset[5])
print(torch.sum(dataset[5]))
print(len(dataset))

# Now when our data is ready, let's define a simple inference model


class SomeInferenceModel(nn.Module):
    """ A simple LSTM model for inferring some property of the data. """

    def __init__(self):
        self.lstm = nn.LSTM()

    def init_state(self):
        h_0 = Variable

    def forward(self, batch):
        pass


