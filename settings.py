import torch

EPOCHS = 300
DATAFILE = "single_review.json"
LEARNING_RATE = 0.01  
GPU = torch.cuda.is_available()
HIDDEN_SIZE = 10
VISUALIZE = True
