# Built in
from inspect import getfullargspec, isclass
import copy

# Third-party
import astropy.coordinates as coord
from astropy.table import Table, QTable
import astropy.units as u
import numpy as np

import gala.dynamics as gd
from gala.dynamics.mockstream import fardal_stream
from gala.integrate.timespec import parse_time_specification
import gala.potential as gp
from gala.units import galactic

# Project
from .stats import ln_normal, ln_normal_ivar, get_ivar
from .track import get_stream_track

__all__ = ['MockStreamModel']

# TODO: add potential_transform and inv_transform methods on MockStreamModel that get called in pack/unpack


class MockStreamModel:
    """ TODO: document this shit

    Parameters
    ----------
    data : `~astropy.table.QTable`, `dict`, or similar
    stream_frame : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
    potential : `~gala.potential.PotentialBase` subclass instance
    """
    potential_cls = None
    potential_units = None

    def __init_subclass__(cls, **kwargs):
        for required in ('potential_cls', 'potential_units'):
            if not getattr(cls, required):
                raise TypeError("Can't instantiate class {cls.__name__} "
                                "without {required} attribute defined"
                                .format(cls=cls, required=required))
        return super().__init_subclass__(**kwargs)

    @u.quantity_input(phi1_0=u.deg, phi1_lim=u.deg, phi1_binsize=u.deg)
    def __init__(self, data, stream_frame,
                 integrate_kw,
                 frozen=None,
                 mockstream_fn=fardal_stream, mockstream_kw=None,
                 galcen_frame=None,
                 phi1_0=0*u.deg,
                 phi1_lim=[-180, 180]*u.deg, phi1_binsize=1*u.deg):

        # Coordinate frame of the stream data
        if not ((isclass(stream_frame) and
                 issubclass(stream_frame, coord.BaseCoordinateFrame)) or
                isinstance(stream_frame, coord.BaseCoordinateFrame)):
            raise TypeError('Invalid stream frame: must either be an astropy '
                            'frame class instance, or the class itself.')

        if (isclass(stream_frame) and
                issubclass(stream_frame, coord.BaseCoordinateFrame)):
            stream_frame = stream_frame()
        self.stream_frame = stream_frame
        self._frame_cls = stream_frame.__class__
        self._frame_comp_names = (
            list(stream_frame.get_representation_component_names().keys()) +
            list(stream_frame.get_representation_component_names('s').keys()))
        self._frame_attrs = stream_frame.frame_attributes

        # Validate the input data
        # TODO: this is broken for distance/distmod...!
        self.data = QTable(data, copy=False) # maintain the original data
        self._data = Table() # a copy with units stripped
        self._data_units = dict()
        self._has_data = dict()
        for i, name in enumerate(self._frame_comp_names):
            self._has_data[name] = name in self.data.colnames
            if (self._has_data[name] and i > 0 and
                    name+'_ivar' not in self.data.colnames):
                raise ValueError("If passing in data for component '{0}', you "
                                 "must also pass in an inverse-variance with "
                                 "name '{0}_ivar' in the input data table"
                                 .format(name))

            if self._has_data[name]:
                self._data_units[name] = self.data[name].unit
                self._data[name] = self.data[name].value

                if i > 0: # skip phi1 or longitude
                    ivar_unit = 1 / self._data_units[name]**2
                    self._data[name+'_ivar'] = \
                        self.data[name+'_ivar'].to_value(ivar_unit)

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
        self.integrate_kw = integrate_kw

        if galcen_frame is None:
            galcen_frame = coord.Galactocentric()
        self.galcen_frame = galcen_frame

        # Auto-validated by quantity_input
        self.phi1_0 = phi1_0
        self.phi1_lim = phi1_lim
        self.phi1_binsize = phi1_binsize

        # Mock-stream generation function:
        self.mockstream_fn = mockstream_fn

        # TODO: some validation of the mockstream arguments?
        if mockstream_kw is None:
            mockstream_kw = dict()
        self.mockstream_kw = mockstream_kw

        # Frozen parameters:
        if frozen is None:
            frozen = dict()
        self.frozen = frozen

        # Compile parameter names: progenitor initial conditions
        self.param_names = dict()
        self.param_names['w0'] = self._frame_comp_names[1:]

    def _unpack_potential_pars(self, potential_cls, p, frozen,
                               fill_frozen=False):
        j = 0
        all_key_vals = dict()

        if isinstance(potential_cls, dict):
            for k in sorted(potential_cls.keys()):
                this_key_vals, this_j = self._unpack_potential_pars(
                    potential_cls[k], p[j:], frozen[k], fill_frozen)
                all_key_vals[k] = this_key_vals
                j += this_j

        else:
            for name in sorted(potential_cls._physical_types.keys()):
                if name in frozen:
                    if fill_frozen:
                        all_key_vals[name] = frozen[name]
                else:
                    all_key_vals[name] = p[j]
                    j += 1

        return all_key_vals, j

    def _pack_potential_pars(self, potential_cls, pars, frozen,
                             fill_frozen=False):
        vals = []

        if isinstance(potential_cls, dict):
            for k in sorted(potential_cls.keys()):
                this_vals = self._pack_potential_pars(potential_cls[k],
                                                      pars[k],
                                                      frozen[k],
                                                      fill_frozen=fill_frozen)
                vals = np.concatenate((vals, this_vals))

        else:
            for name in sorted(potential_cls._physical_types.keys()):
                if name in frozen:
                    val = frozen.get(name, None)
                    if not fill_frozen:
                        continue
                else:
                    val = pars.get(name, None)

                if val is None:
                    raise ValueError("No value passed in for parameter {0}, "
                                     "but it isn't frozen either!".format(k))

                vals.append(val)

        return np.array(vals)

    def pack_pars(self, p, fill_frozen=True):
        vals = []

        if not self.frozen.get('w0', False): # initial conditions
            frozen_w0 = self.frozen.get('w0', dict())
            p_w0 = p.get('w0', dict())
            for k in self.param_names['w0']:
                if k in frozen_w0:
                    val = frozen_w0.get(k, None)
                    if not fill_frozen:
                        continue

                else:
                    val = p_w0.get(k, None)

                if val is None:
                    raise ValueError("No value passed in for parameter {0}, "
                                     "but it isn't frozen either!".format(k))
                vals.append(val)

        if self.frozen.get('potential', False) is True: # potential params
            pot_vals = []

        else: # no potential parameters are frozen
            pot_pars = copy.deepcopy(p['potential'])
            self.potential_transform(pot_pars)
            pot_vals = self._pack_potential_pars(self.potential_cls,
                                                 pot_pars,
                                                 self.frozen['potential'],
                                                 fill_frozen=fill_frozen)

        return np.concatenate((vals, pot_vals))

    def unpack_pars(self, x, fill_frozen=True):
        pars = dict()

        j = 0

        w0_pars = dict()
        if not self.frozen.get('w0', False): # initial conditions
            frozen_w0 = self.frozen.get('w0', dict())
            for name in self.param_names['w0']:
                if name in frozen_w0:
                    if fill_frozen:
                        w0_pars[name] = self.frozen[name]
                else:
                    w0_pars[name] = x[j]
                    j += 1
        pars['w0'] = w0_pars

        if self.frozen.get('potential', False) is True: # potential params
            pot_pars = dict()

        else:
            pot_pars, j = self._unpack_potential_pars(self.potential_cls,
                                                      x[j:],
                                                      self.frozen['potential'],
                                                      fill_frozen=fill_frozen)

            pot_pars = copy.deepcopy(pot_pars) # to protect frozen dict
            self.potential_transform_inv(pot_pars)

        pars['potential'] = pot_pars

        return pars

    # Helper / convenience methods:
    def get_w0(self, **kwargs):
        kw = dict()
        kw['phi1'] = self.phi1_0

        for k, v in kwargs.items():
            kw[k] = v * self._data_units[k] # TODO: distance vs. distmod issues
        kw.update(self._frame_attrs)

        c = self._frame_cls(**kw)
        w0 = gd.PhaseSpacePosition(c.transform_to(self.galcen_frame).data)
        return w0

    def get_hamiltonian(self, **potential_params):
        if isinstance(self.potential_cls, dict):
            pot = gp.CCompositePotential()
            for k in self.potential_cls:
                pot[k] = self.potential_cls[k](units=self.potential_units,
                                               **potential_params[k])
        else:
            pot = self.potential_cls(units=self.potential_units,
                                     **potential_params)
        return gp.Hamiltonian(pot)

    def get_orbit(self, ham, w0):
        try:
            orbit = ham.integrate_orbit(w0, **self.integrate_kw)
        except Exception as e:
            print("Orbit integrate failed: {}".format(str(e)))
            return None

        return orbit

    def get_mockstream(self, ham, orbit):
        if orbit is None:
            return None

        try:
            stream = self.mockstream_fn(ham, orbit, **self.mockstream_kw)
        except TypeError as e:
            print("Mock stream integrate failed: {}".format(str(e)))
            return None

        return stream

    def tracks_ln_likelihood(self, stream):
        stream_c = stream.to_coord_frame(
            self.stream_frame, galactocentric_frame=self.galcen_frame)

        mean_tracks, std_tracks = get_stream_track(
            stream_c, phi1_lim=self.phi1_lim, phi1_binsize=self.phi1_binsize,
            units=self._data_units)

        lls = []
        for name in self._frame_comp_names[1:]: # skip phi1
            ivar = get_ivar(self._data[name+'_ivar'],
                            std_tracks[name](self._data['phi1'])**2)
            ll = ln_normal_ivar(mean_tracks[name](self._data['phi1']),
                                self._data[name], ivar)
            lls.append(ll[np.isfinite(ll)].sum())

        return np.sum(lls)

    def ln_likelihood(self, pars):
        w0 = self.get_w0(**pars['w0'])
        H = self.get_hamiltonian(**pars['potential'])

        orbit = self.get_orbit(H, w0)
        if orbit is None:
            return -np.inf

        if orbit.t[-1] < orbit.t[0]:
            orbit = orbit[::-1]

        stream = self.get_mockstream(H, orbit)
        if stream is None:
            return -np.inf

        return self.tracks_ln_likelihood(stream)

    def w0_ln_prior(self, kw):
        lp = 0.

        # TODO: hack - names assumed below

        # gaussian in phi2
        val = (kw['phi2']*self._data_units['phi2']).to_value(u.deg)
        lp += ln_normal(val, 0., 5.) # MAGIC NUMBER

        # uniform space density from 1 to 100 kpc:
        a, b = 1, 100
        val = (kw['distance']*self._data_units['distance']).to_value(u.kpc)
        lp += np.log(a) + np.log(b) - np.log(b-a) - 2 * np.log(val) # MAGIC NUMBERs

        # gentle gaussian priors in proper motion
        val = (kw['pm_phi1_cosphi2']*self._data_units['pm_phi1_cosphi2']).to_value(u.mas/u.yr)
        lp += ln_normal(val, 0, 25)
        val = (kw['pm_phi2']*self._data_units['pm_phi2']).to_value(u.mas/u.yr)
        lp += ln_normal(val, 0, 25)

        # wide gaussian prior on RV
        val = (kw['radial_velocity']*self._data_units['radial_velocity']).to_value(u.km/u.s)
        lp += ln_normal(val, 0, 350)

        return lp

    def potential_lnprior(self, **_):
        return 0.

    def ln_prior(self, pars):
        lp = 0.
        lp += self.w0_ln_prior(pars['w0'])
        lp += self.potential_ln_prior(pars['potential'])
        return lp

    def ln_posterior(self, p):
        pars = self.unpack_pars(p)

        lp = self.ln_prior(pars)
        if not np.all(np.isfinite(lp)):
            return -np.inf

        ll = self.ln_likelihood(pars)
        if not np.all(np.isfinite(ll)):
            return -np.inf

        return ll + lp

    def __call__(self, p):
        return self.ln_posterior(p)
