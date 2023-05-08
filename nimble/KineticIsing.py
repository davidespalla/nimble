import numpy as np


class KineticIsing():

    def __init__(self, n_units=None, n_timesteps=None):
        self.n_units = n_units
        self.n_timesteps = n_timesteps
        self.h = np.empty((n_units, n_timesteps))
        self.J = np.empty((n_units, n_units))
        self.ll = ki_loglikelihood()

    def fit(self, X):
        pass


def kinietic_ising_loglikelihood(X, h, J):
    theta = np.stack([h for _ in range(len(X))]) + \
        np.einsum('ij,rtj->rtj', J, X)
    ll = np.mean(np.einsum('ijk,ijm->ijkm',
                 X[:, 1:], theta[:, :-1])) - np.mean(np.log(2*np.cosh(theta)))
    return ll


def compute_deltas(X, h, J):
    S = np.mean(X, axis=0)[1:]
    D = np.mean(np.einsum('ijk,ijm->ijkm', X[:, 1:], X[:, :-1]), axis=(0, 1))
    theta = np.stack([h for _ in range(len(X))]) + \
        np.einsum('ij,rtj->rtj', J, X)

    dh = S - np.mean(np.tanh(theta[:, :-1]), axis=0)
    # add a line of zeros at the end of dh for shape consistency. The last h is meaningless
    dh = np.vstack([dh, np.zeros((dh.shape[-1]))])
    dJ = D - np.mean(np.einsum('rti,rtj->rtij',
                     np.tanh(theta[:, :-1]), X[:, :-1]), axis=(0, 1))

    return dh, dJ
