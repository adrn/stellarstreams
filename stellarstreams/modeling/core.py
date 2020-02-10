# Built in
from abc import ABCMeta
from inspect import isclass
import copy
from warnings import warn

# Third-party
import astropy.coordinates as coord
from astropy.table import Table, QTable
import astropy.units as u
import numpy as np

import gala.dynamics as gd
from gala.integrate.timespec import parse_time_specification
import gala.potential as gp
from gala.units import galactic

__all__ = ['BaseStreamModel']

_default_galcen_frame = coord.Galactocentric()
_default_vx_sun = _default_galcen_frame.galcen_v_sun.d_x
_default_vy_sun = _default_galcen_frame.galcen_v_sun.d_y
_default_vz_sun = _default_galcen_frame.galcen_v_sun.d_z


class BaseStreamModel(BaseModel):
    """ TODO: document this shit

    Parameters
    ----------
    data : `~astropy.table.QTable`, `dict`, or similar
    stream_frame : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
    potential : `~gala.potential.PotentialBase` subclass instance
    """

    @u.quantity_input(lon0=u.deg, lon_bins=u.deg)
    def __init__(self, data, stream_frame, potential,
                 frozen=None, lon0=0*u.deg, lon_bins=None):
        super().__init__()

        # TODO: document: 'lon' name because we are agnostic: this could be ICRS, or stream coordinates, or whatever. But the longitude frame component is always taken to be the independent variable

        # Validate the input data
        # TODO: right now, the data must be sky position, distance (not
        # distmod!), proper motions, and radial velocity, but we should also
        # support distance modulus I suppose?
        self.data = QTable(data, copy=False)  # keep the original data table
        self._data = Table()  # a copy with units stripped
        self._data_units = dict()
        self._has_data = dict()  # what data was provided
        for i, name in enumerate(self._frame_comp_names):
            self._has_data[name] = name in self.data.colnames

            if self._has_data[name]:
                self._data_units[name] = self.data[name].unit
                self._data[name] = self.data[name].value

                if i > 0:  # skip ivar for longitude component
                    if name+'_ivar' not in self.data.colnames:
                        warn("No uncertainties provided for component '{0}' - "
                             "if you want to provide uncertainties, you must "
                             "pass in inverse-variance values with name "
                             "'{0}_ivar' in the input data table.".format(name),
                             RuntimeWarning)
                        self._data[name+'_ivar'] = np.zeros(len(self.data))
                        continue

                    # ensure the ivar values are in the same units as the data
                    ivar_unit = 1 / self._data_units[name] ** 2
                    self._data[name+'_ivar'] = \
                        self.data[name+'_ivar'].to_value(ivar_unit)

        # Ensure that the stream_frame is an instance
        if not isinstance(stream_frame, coord.BaseCoordinateFrame):
            raise TypeError('Invalid stream frame input: this must be an '
                            'astropy frame class *instance*.')

        self.stream_frame = stream_frame
        self._frame_cls = stream_frame.__class__
        self._frame_comp_names = (
            list(stream_frame.get_representation_component_names().keys()) +
            list(stream_frame.get_representation_component_names('s').keys()))
        self._frame_attrs = stream_frame.frame_attributes

        # units are auto-validated by quantity_input
        self.lon0 = lon0
        if lon_bins is None:
            lon_bins = np.arange(-180, 180+1e-3, 1.) * u.deg  # default!
        self.lon_bins = lon_bins

        # strip units
        lon_name = self._frame_comp_names[0]
        self._lon0 = lon0.to_value(self._data_units[lon_name])
        self._lon_bins = lon_bins.to_value(self._data_units[lon_name])

        # Frozen parameters:
        if frozen is None:
            frozen = dict()
        self.frozen = frozen

        # TODO: allow also specifying transform functions here?
        self.register_param_group('w0', param_names=self._frame_comp_names[1:],
                                  ln_prior=self.w0_ln_prior)

        self.register_param_group('sun',
                                  param_names=['galcen_distance',
                                               'vx_sun', 'vy_sun', 'vz_sun',
                                               'z_sun'],
                                  ln_prior=self.sun_ln_prior)

        # Deal with the input potential
        # TODO: right now, it must be a composite potential
        if not isinstance(potential, gp.CompositePotential):
            raise ValueError("Invalid input for potential: This must be a "
                             "potential class instance, not '{}'"
                             .format(type(potential)))

        self.potential = potential
        _pot_param_names = {}
        ppars = {}
        for k in sorted(potential.keys()):
            ppars[k] = {}
            for name, val in potential[k].parameters.items():
                ppars[k][name] = val.decompose(potential.units).value
            _pot_param_names[k] = sorted(list(ppars[k].keys()))
        self._in_pot_params = ppars

        self.register_param_group(
            'potential', param_names=_pot_param_names,
            ln_prior=self.potential_ln_prior,
            pack_func=self._pack_potential_pars,
            unpack_finc=self._unpack_potential_pars,
            transform_func=self.potential_transform,
            inv_transform_func=self.potential_transform_inv)

    def _unpack_potential_pars(self, p, frozen, fill_frozen=False):
        j = 0
        key_vals = {}

        if frozen is True:
            # Potential is completely frozen, so use the inputted parameters
            return self._in_pot_params, 0

        for k in sorted(self.potential.keys()):
            key_vals[k] = {}
            for name in sorted(self.potential[k].parameters.keys()):
                if name in frozen and fill_frozen:
                    key_vals[k][name] = frozen[k].get(
                        name, self._in_pot_params[k][name])
                else:
                    key_vals[k][name] = p[j]
                    j += 1

        return key_vals, j

    def _pack_potential_pars(self, pars, frozen, fill_frozen=False):
        vals = []

        if frozen is True and not fill_frozen:
            # Potential is completely frozen and not filling frozen par vals
            return np.array(vals)

        for k in sorted(self.potential.keys()):
            for name in sorted(self.potential[k].parameters.keys()):
                if name in frozen.get(k, {}):
                    val = frozen[k][name]
                    if not fill_frozen:
                        continue
                else:
                    val = pars[k][name]
                vals.append(val)

        return np.array(vals)

    def pack_pars(self, p, fill_frozen=False):
        vals = []

        # Progenitor orbit initial conditions
        if self.frozen.get('w0', False) is not True:
            frozen_w0 = self.frozen.get('w0', dict())  # frozen values
            p_w0 = p.get('w0', dict())  # values passed in via `p['w0']`
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

        # Solar / LSR reference frame
        # TODO: code duplication here relative to the above
        if self.frozen.get('sun', False) is not True:
            frozen_sun = self.frozen.get('sun', dict())  # frozen values
            p_sun = p.get('sun', dict())  # values passed in via `p['sun']`
            for k in self.param_names['sun']:
                if k in frozen_sun:
                    val = frozen_sun.get(k, None)
                    if not fill_frozen:
                        continue

                else:
                    val = p_sun.get(k, None)

                # Here we allow a value to not be specified, then it'll be
                # set to the default
                if val is not None:
                    vals.append(val)

        # Potential
        if self.frozen.get('potential', False) is not True:
            pot_pars = copy.deepcopy(p['potential'])
            self.potential_transform(pot_pars)
            pot_vals = self._pack_potential_pars(
                pot_pars, self.frozen.get('potential', {}),
                fill_frozen=fill_frozen)

            vals = np.concatenate((vals, pot_vals))

        return np.array(vals)

    def unpack_pars(self, x, fill_frozen=True):
        pars = dict()

        j = 0

        w0_pars = dict()
        if self.frozen.get('w0', False) is not True:
            frozen_w0 = self.frozen.get('w0', dict())
            for name in self.param_names['w0']:
                if name in frozen_w0:
                    if fill_frozen:
                        w0_pars[name] = self.frozen[name]
                else:
                    w0_pars[name] = x[j]
                    j += 1
        pars['w0'] = w0_pars

        sun_pars = dict()
        if self.frozen.get('sun', False) is not True: # solar / LSR frame
            frozen_sun = self.frozen.get('sun', dict())
            for name in self.param_names['sun']:
                if name in frozen_sun:
                    if fill_frozen:
                        sun_pars[name] = self.frozen[name]
                else:
                    sun_pars[name] = x[j]
                    j += 1
        pars['sun'] = sun_pars

        pot_pars = dict()
        if self.frozen.get('potential', False) is not True:  # potential params
            pot_pars, j = self._unpack_potential_pars(self.potential_cls,
                                                      x[j:],
                                                      self.frozen['potential'],
                                                      fill_frozen=fill_frozen)

            pot_pars = copy.deepcopy(pot_pars) # to protect frozen dict
            self.potential_transform_inv(pot_pars)

        pars['potential'] = pot_pars

        return pars

    # Helper / convenience methods:
    def get_galcen_frame(self, **kwargs):
        if 'sun' in self.frozen and self.frozen['sun'] is not True:
            frozen = self.frozen['sun']
        else:
            frozen = {}

        vx = kwargs.get('vx_sun', frozen.get('vx_sun', _default_vx_sun.value))
        vy = kwargs.get('vy_sun', frozen.get('vy_sun', _default_vy_sun.value))
        vz = kwargs.get('vz_sun', frozen.get('vz_sun', _default_vz_sun.value))
        print(vx, vy, vz)

        galcen_kwargs = {}
        galcen_kwargs['galcen_v_sun'] = coord.CartesianDifferential(
            [vx, vy, vz] * u.km/u.s)  # TODO: assumed units

        default_dist = frozen.get(
            'galcen_distance',
            _default_galcen_frame.galcen_distance.to_value(u.kpc))
        galcen_kwargs['galcen_distance'] = kwargs.get('galcen_distance',
                                                      default_dist) * u.kpc  # TODO: units

        default_zsun = frozen.get('z_sun',
                                  _default_galcen_frame.z_sun.to_value(u.kpc))
        galcen_kwargs['z_sun'] = kwargs.get('z_sun', default_zsun) * u.kpc # TODO: units

        return coord.Galactocentric(**galcen_kwargs)

    def get_w0(self, galcen_frame, **kwargs):
        kw = dict()
        kw['phi1'] = self.phi1_0

        for k, v in kwargs.items():
            kw[k] = v * self._data_units[k] # TODO: distance vs. distmod issues
        kw.update(self._frame_attrs)

        c = self._frame_cls(**kw)
        w0 = gd.PhaseSpacePosition(c.transform_to(galcen_frame).data)
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

    # To be overridden by the user:
    def potential_transform(self, *args, **kwargs):
        pass

    def potential_transform_inv(self, *args, **kwargs):
        pass

    def w0_ln_prior(self, kw):
        return 0.

    def potential_ln_prior(self, *_, **__):
        return 0.

    def sun_ln_prior(self, *_, **__):
        return 0.
