import astropy.units as u
import gala.potential as gp
from gala.units import galactic
import numpy as np

from ..trackfit import (_pack_potential_pars, _unpack_potential_pars,
                        MockStreamModel)


def test_potential_pack():
    pot1 = gp.HernquistPotential(m=1e5*u.Msun,
                                c=4.23,
                                units=galactic)
    frozen1 = {'c': 1.}
    p1 = [1e6]

    pot2 = gp.MilkyWayPotential()
    frozen2 = {'bulge': {'m': 5e9, 'c': 1},
               'disk': {'a': 3., 'b': 0.2},
               'halo': {'r_s': 15.62},
               'nucleus': {'m': 1.71e9, 'c': 0.07}}
    p2 = [1.1e10, 5.8e11]

    for pot, frozen, p in zip([pot1, pot2],
                              [frozen1, frozen2],
                              [p1, p2]):
        pars_out, j = _unpack_potential_pars(pot, p, frozen=frozen)
        assert j == len(p)
        if isinstance(pot, gp.CompositePotential):
            assert np.allclose(pars_out['disk']['m'], p[0])
            assert np.allclose(pars_out['halo']['m'], p[1])
        else:
            assert np.allclose(pars_out['m'], p[0])

        p_out = _pack_potential_pars(pot, pars_out, frozen=frozen)
        for x, y in zip(p_out, p):
            assert np.allclose(x, y)
        assert len(p) == len(p_out)
