import torch
import models

EPOCHS = 300
DATAFILE = "data/small_data_train.json"
LEARNING_RATE = 0.001
GPU = torch.cuda.is_available()

MODEL = {
    "model": models.SimpleLSTM,
    "hidden_size": 100,
}

HIDDEN_SIZE = 100
VISUALIZE = True
CHECKPOINT_DIR = "checkpoints"
HIST_OPTS = dict(numbins=20,
                 xtickmin=0,
                 xtickmax=6)
