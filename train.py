import torch
from torch.utils.data import DataLoader
import sys
from datasets import YelpReviews
import settings 
import models
from torch.autograd import Variable

dataset = YelpReviews(settings.DATAFILE)
model = models.BaselineModel()
optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
for epoch in range(settings.EPOCHS):
    for feature, target in data_loader:
        feature = Variable(feature.permute(1,0,2))
        target = Variable(target)
        out=model(feature)
        loss = torch.mean((out[-1,0] - target)**2)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("Label: {}".format(target))
        #print("Prediction: {}".format(out))
        print("Loss: {}".format(loss))
	  
