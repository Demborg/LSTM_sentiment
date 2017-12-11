import torch
from torch.utils.data import DataLoader
import sys
from datasets import YelpReviews
import settings 

dataset = YelpReviews(settings.DATAFILE)

data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
for epoch in range(settings.EPOCHS):
    for feature, target in data_loader:
        print(feature)
        print(target)

	  
