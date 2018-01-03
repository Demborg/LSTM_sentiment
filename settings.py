import argparse
import torch
import models
import datasets

# Argument parsing
parser = argparse.ArgumentParser(description="Sentiment analysis through Yelp reviews.")
parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
parser.add_argument('--visualize', action='store_true', help='Enable visdom visualization')
parser.add_argument('--load-path', action='store', help='Path to checkpoint file for evaluation.')
parser.add_argument('--data-path', action='store', help='Path to dataset.')
parser.add_argument('--text', action='store', help='Text for live evaluation.')
args = parser.parse_args()

EPOCHS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 100
GPU = torch.cuda.is_available()

MODEL = {
    "model": models.SimpleLSTM,
    "embedding_dim": 50,
    "hidden_size": 128,
    "num_layers": 1,
    "input_size": 50,
}

DATASET = datasets.GlovePretrained50d
DATA_KWARGS = {
    "glove_path": "glove.6B.50d.txt"
}

VISUALIZE = args.visualize
CHECKPOINT_DIR = "checkpoints"

GPU = torch.cuda.is_available() and args.enable_cuda
HIST_OPTS = dict(numbins=20,
                 xtickmin=0,
                 xtickmax=6)
