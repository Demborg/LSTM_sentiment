import torch
from torch.autograd import Variable

import utils
import settings
import re
import torchwordemb

from live_sentiment import text2vec


if __name__ == "__main__":
    # Load model
    model = utils.generate_model_from_settings()
    utils.load_model_params(model, settings.args.load_path)

    # Load glove
    print("Reading word vectors...")
    vocab, vec = torchwordemb.load_glove_text(settings.DATA_KWARGS["glove_path"])
    print("Done!")

    while True:
        text = input("What's on your mind? \n")
        features = text2vec(text, vocab, vec)
        features = utils.pack_sequence([features])
        (features, lengths) = torch.nn.utils.rnn.pad_packed_sequence(features)
        out = model(features, lengths)

        stars = float(out[0, 0, 0])
        if stars < 1.1:
            print("Watch your language, kid")
        print("Your mind has the following rating: {}".format(stars))

