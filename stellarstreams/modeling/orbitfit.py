# Third-party
import astropy.units as u
import numpy as np

# Project
from .core import BaseStreamModel
from .stats import ln_normal_ivar, get_ivar
from .track import get_orbit_track

__all__ = ['OrbitFitModel']


class OrbitFitModel(BaseStreamModel):
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
                 galcen_frame=None,
                 phi1_0=0*u.deg,
                 phi1_lim=[-180, 180]*u.deg):

        super().__init__(data=data, stream_frame=stream_frame,
                         integrate_kw=integrate_kw, frozen=frozen,
                         galcen_frame=galcen_frame, phi1_0=phi1_0,
                         phi1_lim=phi1_lim)

        # TODO: do something about integrating forward and/or backward...

    def orbit_ln_likelihood(self, orbit):
        orbit_c = orbit.to_coord_frame(
            self.stream_frame, galactocentric_frame=self.galcen_frame)

        mean_tracks = get_orbit_track(orbit_c, phi1_lim=self.phi1_lim,
                                      units=self._data_units)

        lls = []
        for name in self._frame_comp_names[1:]: # skip phi1
            ll = ln_normal_ivar(mean_tracks[name](self._data['phi1']),
                                self._data[name], self._data[name+'_ivar'])
            lls.append(ll[np.isfinite(ll)].sum())

        return np.sum(lls)

    def ln_likelihood(self, pars):
        w0 = self.get_w0(**pars['w0'])
        H = self.get_hamiltonian(**pars['potential'])

        orbit = self.get_orbit(H, w0)
        if orbit is None:
            return -np.inf

        return self.orbit_ln_likelihood(orbit)
