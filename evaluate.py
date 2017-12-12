import sys

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

import datasets
import settings
import models
import utils

"""Evaluates a model on a given dataset
takes two command line parameters, path to saved model and path to dataset"""

# Instansiate dataset
dataset = datasets.YelpReviews(sys.argv[2])
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# Define model and optimizer
model = models.SimpleLSTM(settings.HIDDEN_SIZE)
utils.load_model_params(model, sys.argv[1])


losses = np.zeros(len(dataset))
for i, (feature, target) in enumerate(data_loader):
    # Inference
    feature = Variable(feature.permute(1, 0, 2))
    target = Variable(target)
    out = model(feature)

    # Loss computation and weight update step
    loss = torch.mean((out[-1, 0] - target)**2)
    losses[i] = loss

print("Average loss was: {}".format(np.mean(losses)))





