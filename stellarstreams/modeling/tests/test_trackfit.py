from astropy.table import Table
import astropy.units as u
import gala.coordinates as gc
import gala.dynamics as gd
from gala.dynamics.mockstream import fardal_stream
import gala.potential as gp
from gala.units import galactic
import numpy as np

from ..trackfit import (_pack_potential_pars, _unpack_potential_pars,
                        MockStreamModel)

def get_fake_data(w0, H, fr, size=1):
    data = Table()

    orbit = H.integrate_orbit(w0, dt=-1, n_steps=1000)[::-1]
    stream = fardal_stream(H, orbit, prog_mass=5e4*u.Msun,
                           release_every=4)
    stream_c = stream.to_coord_frame(fr)
    size = len(stream_c)

    data['phi1'] = stream_c.phi1

    data['phi2'] = stream_c.phi2
    data['phi2_ivar'] = 1 / (np.random.normal(1e-3, 1e-5, size) * u.deg)**2

    data['distance'] = stream_c.distance
    data['distance_ivar'] = 1 / (np.random.normal(0.2, 1e-2, size) * u.kpc)**2

    data['pm_phi1_cosphi2'] = stream_c.pm_phi1_cosphi2
    data['pm_phi1_cosphi2_ivar'] = 1 / (np.random.normal(0.1, 1e-2, size) * u.mas/u.yr)**2

    data['pm_phi2'] = stream_c.pm_phi2
    data['pm_phi2_ivar'] = 1 / (np.random.normal(0.1, 1e-2, size) * u.mas/u.yr)**2

    data['radial_velocity'] = stream_c.radial_velocity
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
    true_w0 = gd.PhaseSpacePosition(pos=[10, 0, 0.]*u.kpc,
                                    vel=[0, 200, 0.]*u.km/u.s)
    true_c = true_w0.to_coord_frame(gc.GD1Koposov10)
    pot = gp.HernquistPotential(m=2e11*u.Msun,
                                c=5*u.kpc,
                                units=galactic)

    data = get_fake_data(true_w0, gp.Hamiltonian(pot),
                         gc.GD1Koposov10, size=32)

    model = MockStreamModel(data=data,
                            phi1_0=true_c.phi1,
                            stream_frame=gc.GD1Koposov10,
                            integrate_kw=dict(dt=-0.5, n_steps=2000),
                            potential=pot,
                            frozen={'potential': {'m': 2e11}},
                            mockstream_kw={'prog_mass': 5e4*u.Msun,
                                           'release_every': 1})

    pars = model.pack_pars(
        {'w0': {'phi2': true_c.phi2.degree,
                'pm_phi1_cosphi2': true_c.pm_phi1_cosphi2.value,
                'pm_phi2': true_c.pm_phi2.value,
                'distance': true_c.distance.kpc,
                'radial_velocity': true_c.radial_velocity.to_value(u.km/u.s)},
         'potential': {'c': 5}})
    pars2 = model.unpack_pars(pars)

    w0 = model.get_w0(**pars2['w0'])
    H = model.get_hamiltonian(**pars2['potential'])

    orbit = model.get_orbit(H, w0)
    # orbit.plot()

    stream = model.get_mockstream(H, orbit[::-1])
    # stream.plot()
    # import matplotlib.pyplot as plt
    # plt.show()

    ll = model.tracks_ln_likelihood(stream)
    print(ll)

    ll2 = model.ln_likelihood(pars2)
    lp2 = model(pars)
