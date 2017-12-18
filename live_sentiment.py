import sys

import torch
from torch.autograd import Variable
import numpy as np

import models
import utils
import settings
from colored import fg, bg, stylize 

def get_live_sentiment(model, feature):
    """Takes a trained model and a list of features and returns the
    estimated scores for each timestep"""

    # Inference
    feature = Variable(feature.permute(1, 0, 2))
    out = model(feature)

    return(out)

def rating_to_color(rating):
    """Takes a rating from 0 to 5 and converts that to a grey scale x-term
    safe color"""

    val = int(rating/5. * 23 + 232)
    val = max(0, min(255, val))
    return val

if __name__ == "__main__":
    # load model
    model = models.SimpleLSTM(settings.HIDDEN_SIZE)
    utils.load_model_params(model, sys.argv[1])

    #extract features from string
    features = [ord(c) for c in sys.argv[2]]
    line_array = np.zeros([1, len(features), 256], dtype="float32")
    for i, j in enumerate(features):
        line_array[0,i,j] = 1

    line_array = torch.from_numpy(line_array)
    # print(line_array)

    out = get_live_sentiment(model, line_array)

    #Color stuff
    font_color = fg("#0000ff")

    print("Color range:")
    for i in np.arange(0,5.1,0.5):

        style = bg(rating_to_color(i)) + font_color

        print(stylize(i, style), end='')
    print("\nScored sentence:")

    for i, c in enumerate(sys.argv[2]):
        style = bg(rating_to_color(out[i,0,0])) + font_color

        print(stylize(c, style), end='')

    print()

