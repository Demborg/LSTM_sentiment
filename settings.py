import argparse
import torch
import models

# Argument parsing
parser = argparse.ArgumentParser(description="GPU argument paser")
parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
parser.add_argument('--load-path', action='store', help='Path to checkpoint file for evaluation.')
parser.add_argument('--data-path', action='store', help='Path to dataset.')
parser.add_argument('--text', action='store', help='Text for live evaluation.')
args = parser.parse_args()

EPOCHS = 300
DATAFILE = "data/small_data_train.json"
LEARNING_RATE = 0.001
GPU = torch.cuda.is_available()

MODEL = {
    "model": models.SimpleLSTM,
    "hidden_size": 100,
}

VISUALIZE = True
CHECKPOINT_DIR = "checkpoints"

GPU = torch.cuda.is_available() and args.enable_cuda
HIST_OPTS = dict(numbins=20,
                 xtickmin=0,
                 xtickmax=6)
