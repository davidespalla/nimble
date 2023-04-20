import numpy as np


def codeword_counts(X):
    codewords = {}
    for c in X:
        # binarize population vector and ash it with tobytes()
        ashed_c = (c > 0).astype(int).tobytes()

        # if codeword already there add to frequency
        if ashed_c in codewords.keys():
            codewords[ashed_c] += 1

        # else add codeword to dictionary
        else:
            codewords[ashed_c] = 1

    return codewords


def active_cell_counts(X):
    ks = []
    for c in X:
        ks.append(np.sum((c > 0).astype(int)))
    return np.asarray(ks)


def match_support(x1, x2):
    '''
    Pads shorter array with zeros, to mach size of probabilities for fair entropy calculation
    '''
    if len(x1) > len(x2):
        x2 = np.hstack([x2, np.zeros(len(x1)-len(x2))])
    if len(x2) > len(x1):
        x1 = np.hstack([x1, np.zeros(len(x2)-len(x1))])

    assert len(x1) == len(x2)

    return x1, x2
