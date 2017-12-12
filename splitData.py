import random
from os import path
import sys
random.seed(1337)

def split(filename, test=0.1, validate=0.1):
    """Takes a file and the percentages that should go to test and validation and creates three new files filename_test, filename_train and filename_validation with the given proportion of lines from the original file"""

    if test < 0 or validate < 0 or test + validate > 1:
        raise ValueError('validate and test are the proportions that go to test and validation sets, make sure they are set to sensible values')

    train = 1 - test - validate

    with open(filename, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)

    root, ext = path.splitext(filename)
    offset = 0
    for name, prop in zip(["_train", "_test", "_validate"], [train, test, validate]):
        with open(root+name+ext, 'w') as f:
            stop = int(offset + prop * len(lines) + 1)
            f.writelines(lines[offset:stop])
            offset = stop


if __name__ == "__main__":
    if len(sys.argv) == 2:
        split(sys.argv[1])
    elif len(sys.argv == 4):
        split(sys.arg[1], float(sys.argv[2]), float(sys.argv[3]))
    else:
        print("This takes name of file to split and optionally proportions for test and validation")
