import sys

import torch
from torch.autograd import Variable
import numpy as np

import utils
import settings
from colored import fg, bg, stylize
import re
import torchwordemb

def rating_to_color(rating):
    """Takes a rating from 0 to 5 and converts that to a grey scale x-term
    safe color"""

    val = int(rating/5. * 23 + 232)
    val = max(0, min(255, val))
    return val

def text2vec(text,vocab,vec):
    pattern = re.compile('[^ \w]+')
    features = pattern.sub('', text.lower())

    remapped = []
    for word in features.split(" "):
        if word in vocab:
            remapped.append(vec[vocab[word]])
        else:
            remapped.append(torch.zeros(50))
    features = torch.stack(remapped)

    return Variable(features)


if __name__ == "__main__":
    # load model
    model = utils.generate_model_from_settings()
    utils.load_model_params(model, settings.args.load_path)

    #load glove
    print("Reading word vectors...")
    vocab, vec = torchwordemb.load_glove_text(settings.DATA_KWARGS["glove_path"])
    print("Done!")

    #extract features from string
    features = text2vec(settings.args.text, vocab, vec)
    features = utils.pack_sequence([features])
    (features, lengths) = torch.nn.utils.rnn.pad_packed_sequence(features)
    out = model(features, lengths)

    stars = float(out[0, 0, 0])
    print(stars)

'''
    #Color stuff
    font_color = fg("#0000ff")

    print("Color range:")
    for i in np.arange(0,5.1,0.5):

        style = bg(rating_to_color(i)) + font_color

        print(stylize(i, style), end='')
    print("\nScored sentence:")

    for i, c in enumerate(settings.args.text):
        style = bg(rating_to_color(out[i, 0, 0])) + font_color

        print(stylize(c, style), end='')
'''

