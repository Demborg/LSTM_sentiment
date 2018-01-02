import argparse
import torch
import models
import datasets

# Argument parsing
parser = argparse.ArgumentParser(description="Sentiment analysis through Yelp reviews.")
parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
parser.add_argument('--load-path', action='store', help='Path to checkpoint file for evaluation.')
parser.add_argument('--data-path', action='store', help='Path to dataset.')
parser.add_argument('--text', action='store', help='Text for live evaluation.')
args = parser.parse_args()

EPOCHS = 300
LEARNING_RATE = 0.001
GPU = torch.cuda.is_available()

MODEL = {
    "model": models.EmbeddingGRU,
    "embedding_dim": 8,
    "hidden_size": 100,
    "num_layers": 1,
}

DATASET = datasets.YelpReviewsOneHotChars

VISUALIZE = True
CHECKPOINT_DIR = "checkpoints"

GPU = torch.cuda.is_available() and args.enable_cuda
HIST_OPTS = dict(numbins=20,
                 xtickmin=0,
                 xtickmax=6)
