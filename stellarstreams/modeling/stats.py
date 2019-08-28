# Standard library
import warnings

# Third-party
import numpy as np


def ln_normal(x, mu, std):
    return -0.5 * (x-mu)**2 / std**2 - 0.5*np.log(2*np.pi) - np.log(std)


def ln_normal_ivar(x, mu, ivar):
    return -0.5 * (x-mu)**2 * ivar - 0.5*np.log(2*np.pi) + 0.5*np.log(ivar)


def get_ivar(ivar, extra_var):
    ivar = np.array(ivar)
    extra_var = np.array(extra_var)
    shape = ivar.shape

    ivar = np.atleast_1d(ivar)
    extra_var = np.broadcast_to(np.atleast_1d(extra_var),
                                ivar.shape)

    inf_mask = np.isinf(ivar)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        new_ivar = ivar / (1 + extra_var * ivar)
        new_ivar[inf_mask] = 1 / extra_var[inf_mask]

    return new_ivar.reshape(shape)
