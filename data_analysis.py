from torch.utils.data import DataLoader
from datasets import YelpReviewsOneHotChars
import numpy as np
import settings
from visdom import Visdom
import copy

# Instansiate dataset
dataset = YelpReviewsOneHotChars(settings.DATAFILE)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# Create visdom instance
viz = Visdom()


labels = np.zeros([4, len(dataset)])
for i, (feature, target) in enumerate(data_loader):
    labels[:,i] = target[0]

print(labels)
opts = copy.deepcopy(settings.HIST_OPTS)
opts["title"] = "Real star distribution"
viz.histogram(X=labels[0], opts=opts)
