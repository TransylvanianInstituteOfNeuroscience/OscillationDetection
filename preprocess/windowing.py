import numpy as np


def rolling(a, window):
    # example of call:
    # for jump_index in range(0, len(sig) - BATCH, BATCH):
    #       spikes = rolling(sig[jump_index:jump_index + WAVEFORM_LENGTH + BATCH - 1], WAVEFORM_LENGTH)
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)