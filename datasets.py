import torch
import json
import sys
import numpy as np
import filereader


from torch.utils.data import Dataset


class YelpReviewsOneHotChars(Dataset):
    def __init__(self, path):
        self.reader = filereader.FileReader(path)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, item):
        line = self.reader[item]
        data = json.loads(line)
        features = data["text"]
        features = [ord(c) for c in features]
        line_array = np.zeros([len(features), 256], dtype="float32")
        for i, j in enumerate(features):
            line_array[i, j] = 1
        keys = ["stars", "useful", "cool", "funny"]
        targets = np.array([float(data[i]) for i in keys], dtype='float32')

        return torch.from_numpy(line_array), torch.from_numpy(targets)


class YelpReviewsCharIdxes(Dataset):
    """ Dataset built for loading yelp reviews into an indexed format.
    Should be used together with models that learn their own embeddings.
    """
    def __init__(self, path):
        self.reader = filereader.FileReader(path)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, item):
        line = self.reader[item]
        data = json.loads(line)
        features = data["text"]
        features = np.array([ord(c) for c in features], dtype="int64")
        keys = ["stars", "useful", "cool", "funny"]
        targets = np.array([float(data[i]) for i in keys], dtype='float32')

        return torch.from_numpy(features), torch.from_numpy(targets)


class RandomData(Dataset):
    """ Random data generator, designed for speed checks """

    def __init__(self, path, output_len):
        self.len = 0
        self.output_len = output_len
        print("Dataset: going through all reviews...")
        with open(path) as f:
            for line in f:
                self.len += 1
        print("Dataset: Is ready!")

        self.file = IndexedOpen(path)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return torch.randn(self.output_len, 256), torch.ones(4)


if __name__ == "__main__":
    dataset = YelpReviewsOneHotChars(sys.argv[1])
    print(len(dataset))
    print(dataset[1])
    import code
    code.interact(local=locals())
                

