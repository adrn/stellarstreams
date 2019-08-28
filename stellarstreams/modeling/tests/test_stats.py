# Third-party
import numpy as np
from scipy.stats import norm

# Package
from ..stats import ln_normal, ln_normal_ivar, get_ivar

size = 128


def test_ln_normal():
    x = np.random.uniform(-10, 10, size=size)
    mu = np.random.uniform(-10, 10, size=size)
    std = np.random.uniform(1e-3, 2, size=size)

    assert np.allclose(ln_normal(x, mu, std),
                       norm.logpdf(x, mu, std))


def test_ln_normal_ivar():
    x = np.random.uniform(-10, 10, size=size)
    mu = np.random.uniform(-10, 10, size=size)
    std = np.random.uniform(1e-3, 2, size=size)

    assert np.allclose(ln_normal_ivar(x, mu, 1 / std**2),
                       norm.logpdf(x, mu, std))


def test_get_ivar():
    # Array input:
    ivar = np.array([1., 0., np.inf])
    extra_var = np.ones_like(ivar)
    new_ivar = get_ivar(ivar, extra_var)
    expected = [0.5, 0, 1.]

    for i in range(len(ivar)):
        assert np.isclose(new_ivar[i], expected[i])

    # Mixed input:
    new_ivar = get_ivar(ivar, 1.)
    for i in range(len(ivar)):
        assert np.isclose(new_ivar[i], expected[i])

    # Scalar input:
    for n, iv in enumerate(ivar):
        new_ivar = get_ivar(iv, extra_var[n])
        assert np.isclose(new_ivar, expected[n])
