import torch
from torch.utils.data import DataLoader
import sys
from datasets import YelpReviews
import settings
import models
from torch.autograd import Variable
from visdom import Visdom
import numpy as np

# Instansiate dataset
dataset = YelpReviews(settings.DATAFILE)

# Define model and optimizer
# model = models.BaselineModel(settings.HIDDEN_SIZE)
model = models.SimpleLSTM(settings.HIDDEN_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

# Visualization thorugh visdom
viz = Visdom()
loss_plot = viz.line(X=np.array([0]), Y=np.array([0]))
dist_hist = viz.histogram(X=np.random.rand(100), opts=dict(numbins=20))

data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

counter = 0
for epoch in range(settings.EPOCHS):
    stars = np.zeros(len(dataset))
    for i, (feature, target) in enumerate(data_loader):
        feature = Variable(feature.permute(1,0,2))
        target = Variable(target)
        out=model(feature)
        stars[i] = out[-1,0,0]
        loss = torch.mean((out[-1,0] - target)**2)
        viz.line(win=loss_plot, X=np.array([counter]), Y=loss.data.numpy(), update='append')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("Label: {}".format(target))
        # print("Prediction: {}".format(out[-1]))
        # print("Loss: {}".format(loss))

        counter+=1
    viz.histogram(win=dist_hist, X=stars, opts=dict(numbins=20))
    print(stars)

