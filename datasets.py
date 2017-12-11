import torch
import json

from torch.utils.data import Dataset

class YelpReviews(Dataset):
    def __init__(self, path):
        self.file = path
        self.len = 0
        with open(path) as f:
            for line in f:
                self.len += 1

    def __len__(self):
        return self.len

if __name__ == "__main__":
    dataset = YelpReviews("small_data.json")
    print(len(dataset))
                

