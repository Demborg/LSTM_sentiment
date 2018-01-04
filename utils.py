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
    path = os.path.join(os.getcwd(), name)
    model.load_state_dict(torch.load(path))


def generate_model_from_settings():
    """ Uses the information in the settings.MODEL to generate a model """
    return settings.MODEL["model"](**settings.MODEL)


def pad_sequence(sequences, batch_first=False):
    r"""Pad a list of variable length Variables with zero

    ``pad_sequence`` stacks a list of Variables along a new dimension,
    and padds them to equal length. For example, if the input is list of
    sequences with size ``Lx*`` and if batch_first is False, and ``TxBx*``
    otherwise. The list of sequences should be sorted in the order of
    decreasing length.

    B is batch size. It's equal to the number of elements in ``sequences``.
    T is length longest sequence.
    L is length of the sequence.
    * is any number of trailing dimensions, including none.

    Note:
        This function returns a Variable of size TxBx* or BxTx* where T is the
            length of longest sequence.
        Function assumes trailing dimensions and type of all the Variables
            in sequences are same.

    Arguments:
        sequences (list[Variable]): list of variable length sequences.
        batch_first (bool, optional): output will be in BxTx* if True, or in
            TxBx* otherwise

    Returns:
        Variable of size ``T x B x * `` if batch_first is False
        Variable of size ``B x T x * `` otherwise
    """

    # assuming trailing dimensions and type of all the Variables
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_variable = torch.autograd.Variable(sequences[0].data.new(*out_dims).zero_())
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
                raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the variable
        if batch_first:
            out_variable[i, :length, ...] = variable
        else:
            out_variable[:length, i, ...] = variable

    return out_variable


def pack_sequence(sequences):
    r"""Packs a list of variable length Variables

    ``sequences`` should be a list of Variables of size ``Lx*``, where L is
    the length of a sequence and * is any number of trailing dimensions,
    including zero. They should be sorted in the order of decreasing length.

    Arguments:
        sequences (list[Variable]): A list of sequences of decreasing length.

    Returns:
        a :class:`PackedSequence` object
    """
    return torch.nn.utils.rnn.pack_padded_sequence(pad_sequence(sequences), [v.size(0) for v in sequences])


def collate_to_packed(batch):
    '''Collates list of samples to minibatch'''

    batch = sorted(batch, key=lambda item: -len(item[0]))
    features = [i[0] for i in batch]
    targets = torch.stack([i[1] for i in batch])

    features = pack_sequence(features)
    features, lengths = torch.nn.utils.rnn.pad_packed_sequence(features, padding_value=0)

    return features, lengths, targets