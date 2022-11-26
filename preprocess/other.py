import numpy as np


def scale_factor(snr, signal, noise):
    return np.sqrt(snr * np.sum(np.square(noise)) / np.sum(np.square(signal)))


def drown_signal(snr, signal, noise):
    sf = scale_factor(snr, signal, noise)
    return sf*signal + noise