import torch

EPOCHS = 300
DATAFILE = "small_data.json"
LEARNING_RATE = 0.001
GPU = torch.cuda.is_available()
HIDDEN_SIZE = 100
VISUALIZE = True
HIST_OPTS = dict(numbins=20,
                 xtickmin=0,
                 xtickmax=6)
