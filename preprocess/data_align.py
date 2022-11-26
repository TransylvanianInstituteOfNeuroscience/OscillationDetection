import numpy as np

from amplitude_thresholding.nn.flow_constants import WAVEFORM_OFFSET, WAVEFORM_LENGTH


def align(signal, timestamps, waveforms):
    peak_ind = np.argmin(waveforms, axis=1)

    timestamps = timestamps - (WAVEFORM_OFFSET - peak_ind)
    timestamps = timestamps.astype(int)

    waveforms = []
    for ts in timestamps:
        waveforms.append(signal[ts: ts + WAVEFORM_LENGTH])
    waveforms = np.array(waveforms)

    return timestamps, waveforms