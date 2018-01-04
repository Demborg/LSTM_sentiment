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
dataset = settings.DATASET(settings.args.data_path)
data_loader = DataLoader(dataset, batch_size=settings.BATCH_SIZE,
                         shuffle=True, num_workers=4, collate_fn=utils.collate_to_packed)

# Define model and optimizer
model = utils.generate_model_from_settings()
utils.load_model_params(model, settings.args.load_path)


losses = np.zeros(len(dataset))
for i, (feature, lengths, target) in enumerate(data_loader):
    # Inference
    out = model(feature, lengths)

    # Loss computation and weight update step
    loss = torch.mean((out[-1, :, 0] - target[:, 0])**2)
    losses[i] = loss

print("Average loss was: {}".format(np.mean(losses)))





