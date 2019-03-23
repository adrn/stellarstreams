# Built in
from inspect import getfullargspec

# Third-party
import astropy.coordinates as coord
from astropy.table import Table
import astropy.units as u
import numpy as np

from gala.dynamics.mockstream import fardal_stream
from gala.integrate.timespec import parse_time_specification
import gala.potential as gp
from gala.units import galactic

# Project
from .stats import ln_normal, ln_normal_ivar
from .track import get_stream_track

__all__ = ['MockStreamModel']


# TODO: should be able to do frozen=dict(potential=True) and have it auto-fill the values from the potential instance passed in
def _unpack_potential_pars(potential, p, frozen):
    j = 0
    all_key_vals = dict()
    if isinstance(potential, gp.CompositePotential):
        # Have to deal with the fact that there are multiple potential keys
        for k in sorted(potential.keys()):
            this_frozen = frozen[k]
            key_vals = []
            for name in sorted(potential[k].parameters.keys()):
                if name in this_frozen:
                    key_vals.append((name, this_frozen[name]))
                else:
                    key_vals.append((name, p[j]))
                    j += 1
            all_key_vals[k] = dict(key_vals)

    else:
        for name in sorted(potential.parameters.keys()):
            if name in frozen:
                all_key_vals[name] = frozen[name]
            else:
                all_key_vals[name] = p[j]
                j += 1

    return all_key_vals, j


def _pack_potential_pars(potential, pars, frozen, fill_frozen=False):
    vals = []
    if isinstance(potential, gp.CompositePotential):
        # Have to deal with the fact that there are multiple potential keys
        for k in sorted(potential.keys()):
            this_frozen = frozen[k]
            for name in sorted(potential[k].parameters.keys()):
                if name in this_frozen:
                    val = this_frozen.get(name, None)
                    if not fill_frozen:
                        continue
                else:
                    val = pars[k].get(name, None)

                if val is None:
                    raise ValueError("No value passed in for parameter {0}, but "
                                     "it isn't frozen either!".format(k))

                vals.append(val)

    else:
        for name in sorted(potential.parameters.keys()):
            if name in frozen:
                val = frozen.get(name, None)
                if not fill_frozen:
                    continue
            else:
                val = pars.get(name, None)

            if val is None:
                raise ValueError("No value passed in for parameter {0}, but "
                                 "it isn't frozen either!".format(k))

            vals.append(val)

    return np.array(vals)


class MockStreamModel:
    """ TODO: document this shit

    Parameters
    ----------
    data : `~astropy.table.Table`, `dict`, or similar
    stream_frame : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
    potential : `~gala.potential.PotentialBase` subclass instance
    """

    @u.quantity_input(phi1_0=u.deg, phi1_lim=u.deg, phi1_binsize=u.deg)
    def __init__(self, data, stream_frame,
                 integrate_kw,
                 potential, potential_lnprior=None,
                 mockstream_fn=fardal_stream, mockstream_kw=None,
                 galcen_frame=None,
                 phi1_0=0*u.deg,
                 phi1_lim=[-180, 180]*u.deg, phi1_binsize=1*u.deg):

        # Coordinate frame of the stream data
        if not (issubclass(stream_frame, coord.BaseCoordinateFrame) or
                isinstance(stream_frame, coord.BaseCoordinateFrame)):
            raise TypeError('Invalid stream frame: must either be an astropy '
                            'frame class instance, or the class itself.')

        if issubclass(stream_frame, coord.BaseCoordinateFrame):
            stream_frame = stream_frame()
        self.stream_frame = stream_frame
        self._frame_cls = stream_frame.__class__
        self._frame_comp_names = (
            list(stream_frame.get_representation_component_names().keys()) +
            list(stream_frame.get_representation_component_names('s').keys()))
        self._frame_attrs = stream_frame.frame_attributes

        # Validate the input data
        self.data = Table(data)

        self._has_data = dict()
        for name in self._frame_comp_names:
            self._has_data[name] = name in self.data.colnames
            if self._has_data[name] and name+'_ivar' not in self.data.colnames:
                raise ValueError("If passing in data for component '{0}', you "
                                 "must also pass in an inverse-variance with "
                                 "name '{0}_ivar' in the input data table"
                                 .format(name))

        # Validate integrate_kw arguments
        valid_integrate_args = getfullargspec(parse_time_specification).args
        valid_name_kw = dict()
        for k in integrate_kw:
            if k in valid_integrate_args[1:]: # skip "units"
                valid_name_kw[k] = integrate_kw[k]
        units = galactic # note: this doesn't actually matter
        try:
            _test_times = parse_time_specification(units=units, **valid_name_kw)
        except ValueError:
            raise ValueError('Invalid orbit integration time information '
                             'specified: you passed in "{}", but we need '
                             'a full specification of the orbit time grid as '
                             'required by '
                             'gala.integrate.parse_time_specification'
                             .format(integrate_kw))

        # TODO: deal with potential / ln_prior
        # Validate specification of the potential and integration parameters
        if not (issubclass(potential, gp.PotentialBase) or
                isinstance(potential, gp.PotentialBase)):
            raise TypeError('Invalid potential object: must either be a gala '
                            'potential class instance, or the class itself.')

        if galcen_frame is None:
            galcen_frame = coord.Galactocentric()
        self.galcen_frame = galcen_frame

        # Auto-validated by quantity_input
        self.phi1_0 = phi1_0
        self.phi1_lim = phi1_lim
        self.phi1_binsize = phi1_binsize

        # TODO: some validation of the mockstream arguments
        if mockstream_kw is None:
            mockstream_kw = dict()
        mockstream_kw = dict()

        # Compile parameter names: progenitor initial conditions
        self.param_names = ['{}_0' for x in self._frame_comp_names[1:]]

    def pack_pars(self, p, fill_frozen=True):
        vals = []
        for k in self.param_names:
            if k in self.frozen:
                val = self.frozen.get(k, None)
                if not fill_frozen:
                    continue

            else:
                val = p.get(k, None)

            if val is None:
                raise ValueError("No value passed in for parameter {0}, but "
                                 "it isn't frozen either!".format(k))
            vals.append(val)

        if self.frozen['potential'] is True: # all potential params frozen
            pot_vals = []

        else: # no potential parameters are frozen
            pot_vals = _pack_potential_pars(potential=self.potential,
                                            pars=p['potential'],
                                            frozen=self.frozen['potential'],
                                            fill_frozen=fill_frozen)

        return np.concatenate((vals, pot_vals))

    def unpack_pars(self, x):
        pars = dict()

        j = 0
        for name in self.param_names:
            if name in self.frozen:
                pars[name] = self.frozen[name]
            else:
                pars[name] = x[j]
                j += 1

        if self.frozen['potential'] is True: # all potential params frozen
            pot_pars = dict()

        else:
            pot_pars, j = _unpack_potential_pars(self.potential, x[j:],
                                                 self.frozen['potential'])
        pars['potential'] = pot_pars

        return pars

    def get_w0(self, p):
        phi2, d, pm1, pm2, rv, lnM, c = p
        c = self.stream_frame.__class__(phi1=self.phi1_0,
                                        phi2=phi2*u.deg,
                                        distance=d*u.kpc,
                                        pm_phi1_cosphi2=pm1*u.mas/u.yr,
                                        pm_phi2=pm2*u.mas/u.yr,
                                        radial_velocity=rv*u.km/u.s,
                                        **self.stream_frame.frame_attributes)
        w0 = gd.PhaseSpacePosition(c.transform_to(self.galcen_frame).data)
        return w0

    def get_orbit(self, p):
        M = np.exp(p[-2]) * u.Msun
        c = p[-1]

        ham = gp.Hamiltonian(gp.MilkyWayPotential(halo=dict(m=M, a=1., b=1., c=c)))

        c = self.get_w0(p)

        try:
            orbit = ham.integrate_orbit(w0, dt=-np.abs(self.dt),
                                        n_steps=self.n_steps)[::-1]
        except:
            return None

        return orbit, ham

    def get_mockstream(self, p):
        orbit, ham = self.get_orbit(p)

        if orbit is None:
            return None

        # TODO: vary mass-loss!
        n_times = len(orbit.t)
        prog_mass = np.linspace(5e4, 1e0, n_times) * u.Msun
        prog_mass[-300:] = 1e0 * u.Msun

        try:
            stream = mockstream.dissolved_fardal_stream(ham, orbit, prog_mass,
                                                        t_disrupt=0*u.Gyr,
                                                        release_every=self.release_every,
                                                        seed=42)
        except TypeError:
            return None

        return stream

    def tracks_ln_likelihood(self, stream):
        tracks = get_stream_track(stream.to_coord_frame(self.stream_frame,
                                                        galactocentric_frame=self.galcen_frame),
                                  stream.to_coord_frame(coord.ICRS,
                                                        galactocentric_frame=self.galcen_frame),
                                  phi1_lim=self.phi1_lim,
                                  phi1_binsize=self.phi1_binsize)

        # _grid = np.linspace(-100, 20, 1024)
        # for k in tracks:
        #     plt.figure(figsize=(10, 5))
        #     plt.scatter(data['phi1'], data[k], zorder=-100)
        #     plt.plot(_grid, tracks[k](_grid), marker='')

        # TODO:
        extra_var = dict()
        extra_var['phi2'] = 0.1 ** 2
        extra_var['distmod'] = 0.02 ** 2
        extra_var['pmra'] = 0.1 ** 2
        extra_var['pmdec'] = 0.1 ** 2
        extra_var['rv'] = 5. ** 2

        lls = []
        for name in ['phi2', 'distmod', 'pmra', 'pmdec', 'rv']:
            ll = ln_normal_ivar(tracks[name](data['phi1']),
                                data[name],
                                get_ivar(data[name+'_ivar'], extra_var[name]))
            lls.append(ll[np.isfinite(ll)].sum())

        return np.sum(lls)

    def ln_likelihood(self, p):
        # TODO: vary potential?
        # ham = self.ham
        stream = self.get_mockstream(p)

        if stream is None:
            return -np.inf

        return self.tracks_ln_likelihood(stream)

    def ln_prior(self, p):
        phi2, d, pm1, pm2, rv, lnM, c = p

        lp = 0.

        # TODO HACK: hard-coded
        lp += ln_normal(phi2, 0., 3.)
        lp += ln_normal(d, 8., 4.)
        lp += ln_normal(pm1, 0, 50.)
        lp += ln_normal(pm2, 0, 50.)
        lp += ln_normal(rv, 0, 400.)

        lp += ln_normal(lnM, 27., 0.5)
        lp += ln_normal(c, 1, 0.2)

        return lp

    def ln_posterior(self, p):
        lp = self.ln_prior(p)
        if not np.all(np.isfinite(lp)):
            return -np.inf

        ll = self.ln_likelihood(p)
        if not np.all(np.isfinite(ll)):
            return -np.inf

        return ll + lp

    def __call__(self, p):
        return self.ln_posterior(p)
