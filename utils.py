import torch
import os
import settings


def save_model_params(model, name):
    """ Saves the parameters of the specified model.
    :param model: Model to use.
    :param name: Extra name used to namespace the model.
    :return: Nothing
    """
    path = os.path.join(os.getcwd(), settings.CHECKPOINT_DIR, name)
    torch.save(model.state_dict(), path)


def load_model_params(model, name):
    """ Loads parameters for specified model. Analogous to save_model_params() """
    path = os.path.join(os.getcwd(), settings.CHECKPOINT_DIR, name)
    model.load_state_dict(torch.load(path))
