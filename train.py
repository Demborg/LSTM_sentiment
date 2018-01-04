import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from visdom import Visdom

import settings
import utils

# Instansiate dataset
dataset = settings.DATASET(settings.args.data_path, **settings.DATA_KWARGS)
data_loader = DataLoader(dataset, batch_size=settings.BATCH_SIZE,
                         shuffle=True, num_workers=4, collate_fn=utils.collate_to_packed)

# Define model and optimizer
model = utils.generate_model_from_settings()
optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

# Log file is namespaced with the current model
log_file = "logs/{}_{}.csv".format(model.get_name(), settings.args.data_path.split("/")[-1].split(".json")[0])

if settings.VISUALIZE:
    # Visualization thorugh visdom
    viz = Visdom()
    loss_plot = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(showlegend=True, title="Loss"))
    hist_opts = settings.HIST_OPTS
    hist_opts["title"] = "Predicted star distribution"
    dist_hist = viz.bar(X=np.array([0, 0, 0]), opts=dict(title="Predicted stars"))
    real_dist_hist = viz.bar(X=np.array([0, 0, 0]))

# Move stuff to GPU
if settings.GPU:
    data_loader.pin_memory = True
    model.cuda()

if settings.VISUALIZE:
    smooth_loss = 7 #approx 2.5^2
    decay_rate = 0.99
    smooth_real_dist = np.array([0, 0, 0, 0, 0], dtype=float)
    smooth_pred_dist = np.array([0, 0, 0, 0, 0], dtype=float)

    counter = 0

for epoch in range(settings.EPOCHS):

    # Main train loop
    length = len(dataset)/settings.BATCH_SIZE
    print("Starting epoch {} with length {}".format(epoch, length))
    for i, (feature, lengths, target) in enumerate(data_loader):
        if settings.GPU:
            feature = feature.cuda(async=True)
            target = target.cuda(async=True)

        out = model(feature, lengths)

        # Loss computation and weight update step
        loss = torch.mean((out[0, :, 0] - target[:, 0])**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss
        with open(log_file, 'a') as logfile:
            logfile.write("{},".format(float(loss)))

        # Visualization update
        if settings.VISUALIZE:
            smooth_loss = smooth_loss * decay_rate + (1-decay_rate) * loss.data.cpu().numpy()
            viz.updateTrace(win=loss_plot, X=np.array([counter]), Y=loss.data.cpu().numpy(), name='loss')
            viz.updateTrace(win=loss_plot, X=np.array([counter]), Y=smooth_loss, name='smooth loss')
            real_star = target[:, 0].data.cpu().numpy().astype(int)
            pred_star = out[0, :, 0].data.cpu().numpy().round().clip(1,5).astype(int)
            for idx in range(len(real_star)):
                smooth_pred_dist[pred_star[idx]-1] += 1
                smooth_real_dist[real_star[idx]-1] += 1
            smooth_real_dist *= decay_rate
            smooth_pred_dist *= decay_rate

            viz.bar(win=dist_hist, X=smooth_pred_dist)
            viz.bar(win=real_dist_hist, X=smooth_real_dist)

            counter += 1

        # Progress update
        if i % 10 == 0:
            sys.stdout.write("\rIter {}/{}, loss: {}".format(i, length, float(loss)))
    print("Epoch finished with last loss: {}".format(float(loss)))

    # Visualize distribution and save model checkpoint
    name = "{}_epoch{}.params".format(model.get_name(), epoch)
    utils.save_model_params(model, name)
    print("Saved model params as: {}".format(name))



