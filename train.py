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


def my_collate(batch):
    '''Collates list of samples to minibatch'''

    batch = sorted(batch, key=lambda item: -len(item[0]))
    features = [i[0] for i in batch]
    targets = torch.stack([i[1] for i in batch])

    features = utils.pack_sequence(features)

    return features, targets



# Instansiate dataset
dataset = settings.DATASET(settings.args.data_path)
data_loader = DataLoader(dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=1, collate_fn=my_collate)

# Define model and optimizer
model = utils.generate_model_from_settings()
optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

# Visualization thorugh visdom
viz = Visdom()
loss_plot = viz.line(X=np.array([0]), Y=np.array([0]))
hist_opts = settings.HIST_OPTS
hist_opts["title"] = "Predicted star distribution"
dist_hist = viz.histogram(X=np.random.rand(100), opts=hist_opts)

# Move stuff to GPU
if settings.GPU:
    data_loader.pin_memory = True
    model.cuda()

counter = 0
for epoch in range(settings.EPOCHS):
    # Stars for histogram
    stars = np.zeros(len(dataset))

    # Main epoch loop
    length = len(dataset)/settings.BATCH_SIZE
    print("Starting epoch {} with length {}".format(epoch, length))
    for i, (feature, target) in enumerate(data_loader):
        if settings.GPU:
            feature = feature.cuda(async=True)
            target = target.cuda(async=True)

        out = model(feature)

        # Loss computation and weight update step
        loss = torch.mean((out - target)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Visualization update
        stars[i] = out[-1, 0, 0]
        viz.line(win=loss_plot, X=np.array([counter]), Y=loss.data.cpu().numpy(), update='append')
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



