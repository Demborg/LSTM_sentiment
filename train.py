import sys

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from visdom import Visdom
import numpy as np

import datasets
import settings
import models
import utils


# Instansiate dataset
dataset = datasets.YelpReviews(settings.DATAFILE)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# Define model and optimizer
model = utils.generate_model_from_settings()
optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

# Visualization thorugh visdom
viz = Visdom()
loss_plot = viz.line(X=np.array([0]), Y=np.array([0]))
hist_opts = settings.HIST_OPTS
hist_opts["title"] = "Predicted star distribution"
dist_hist = viz.histogram(X=np.random.rand(100), opts=hist_opts)

counter = 0
for epoch in range(settings.EPOCHS):
    # Stars for histogram
    stars = np.zeros(len(dataset))

    # Main epoch loop
    length = len(dataset)
    print("Starting epoch {} with length {}".format(epoch, length))
    for i, (feature, target) in enumerate(data_loader):
        # Inference
        feature = Variable(feature.permute(1, 0, 2))
        target = Variable(target)
        out = model(feature)

        # Loss computation and weight update step
        loss = torch.mean((out[-1, 0] - target)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Visualization update
        stars[i] = out[-1, 0, 0]
        viz.line(win=loss_plot, X=np.array([counter]), Y=loss.data.numpy(), update='append')
        counter += 1

        # Progress update
        if i % 10 == 0:
            sys.stdout.write("\rIter {}/{}, loss: {}".format(i, length, float(loss)))
    print("Epoch finished with last loss: {}".format(float(loss)))

    # Visualize distribution and save model checkpoint
    viz.histogram(win=dist_hist, X=stars, opts=hist_opts)
    name = "{}_epoch{}.params".format(model.get_name(), epoch)
    utils.save_model_params(model, name)
    print("Saved model params as: {}".format(name))


