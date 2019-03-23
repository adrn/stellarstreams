# Third-party
import numpy as np

def ln_normal(x, mu, std):
    return -0.5 * (x-mu)**2 / std**2 - 0.5*np.log(2*np.pi) - np.log(std)

def ln_normal_ivar(x, mu, ivar):
    return -0.5 * (x-mu)**2 * ivar - 0.5*np.log(2*np.pi) + 0.5*np.log(ivar)

def get_ivar(ivar, extra_var):
    return ivar / (1 + extra_var * ivar)
