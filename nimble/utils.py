import numpy as np


def zeropad_arrays(arrays):
    '''
    Takes a list of arrays, zero-pads all arrays to the right to match longest array in length.
    Returns list of arrays with matched length.
    '''
    padded = []
    max_len = np.max([len(x) for x in arrays])
    for x in arrays:
        padded.append(np.hstack([x, np.zeros(max_len-len(x))]))

    return padded
