import torch
import json
import sys

from torch.utils.data import Dataset

class YelpReviews(Dataset):
    def __init__(self, path):
        self.file = path
        self.len = 0
        print("Dataset: going through all reviews...")
        with open(path) as f:
            for line in f:
                self.len += 1
        print("Dataset: Is ready!")

    def __len__(self):
        return self.len

if __name__ == "__main__":
    dataset = YelpReviews(sys.argv[1])
    print(len(dataset))
                

