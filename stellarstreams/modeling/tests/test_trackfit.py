from astropy.table import Table
import astropy.units as u
import gala.coordinates as gc
import gala.potential as gp
from gala.units import galactic
import numpy as np

from ..trackfit import (_pack_potential_pars, _unpack_potential_pars,
                        MockStreamModel)

def get_fake_data(size=1, icrs=False):
    data = Table()

    if icrs:
        pass

    else:
        data['phi1'] = np.random.uniform(-25, 25, size) * u.deg

        data['phi2'] = np.random.uniform(-2, 2, size) * u.deg
        data['phi2_ivar'] = 1 / (np.random.normal(1e-3, 1e-5, size) * u.deg)**2

        data['distmod'] = np.random.uniform(14, 16, size) * u.mag
        data['distmod_ivar'] = 1 / (np.random.normal(0.2, 1e-2, size) * u.mag)**2

        data['pm_phi1_cosphi2'] = np.random.uniform(5, 6, size) * u.mas/u.yr
        data['pm_phi1_cosphi2_ivar'] = 1 / (np.random.normal(0.1, 1e-2, size) * u.mas/u.yr)**2

        data['pm_phi2'] = np.random.uniform(-1, 1, size) * u.mas/u.yr
        data['pm_phi2_ivar'] = 1 / (np.random.normal(0.1, 1e-2, size) * u.mas/u.yr)**2

        data['radial_velocity'] = np.random.uniform(100, 250, size) * u.km/u.s
        data['radial_velocity_ivar'] = 1 / (np.random.normal(1, 1e-1, size) * u.km/u.s)**2

    return data


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


def test_mockstreammodel_gd1():
    # Make some fake data:
    data = get_fake_data(size=32)
    pot = gp.HernquistPotential(m=1e10*u.Msun,
                                c=10*u.kpc,
                                units=galactic)

    model = MockStreamModel(data=data,
                            stream_frame=gc.GD1Koposov10,
                            integrate_kw=dict(dt=0.5, n_steps=2000),
                            potential=pot,
                            frozen={'potential': {'m': 1e10}},
                            mockstream_kw={'prog_mass': 5e4*u.Msun,
                                           'release_every': 1})
    pars = model.pack_pars({'w0': {'phi2': 0.,
                                   'pm_phi1_cosphi2': 5.,
                                   'pm_phi2': 0.,
                                   'distance': 10.,
                                   'radial_velocity': 150.},
                            'potential': {'c': 10}})
    pars2 = model.unpack_pars(pars)
    print(pars2)
