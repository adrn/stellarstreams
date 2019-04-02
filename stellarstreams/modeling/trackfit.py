# Third-party
import astropy.units as u
import numpy as np
from gala.dynamics.mockstream import fardal_stream

# Project
from .core import BaseStreamModel
from .stats import ln_normal_ivar, get_ivar
from .track import get_stream_track

__all__ = ['BaseMockStreamModel']


class BaseMockStreamModel(BaseStreamModel):
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
                 phi1_0=0*u.deg,
                 phi1_lim=[-180, 180]*u.deg, phi1_binsize=1*u.deg):

        super().__init__(data=data, stream_frame=stream_frame,
                         integrate_kw=integrate_kw, frozen=frozen,
                         phi1_0=phi1_0, phi1_lim=phi1_lim)

        # Mock-stream generation function:
        self.mockstream_fn = mockstream_fn

        # TODO: some validation of the mockstream arguments?
        if mockstream_kw is None:
            mockstream_kw = dict()
        self.mockstream_kw = mockstream_kw

        # Auto-validated by quantity_input
        self.phi1_binsize = phi1_binsize

    def get_mockstream(self, ham, orbit):
        if orbit is None:
            return None

        try:
            stream = self.mockstream_fn(ham, orbit, **self.mockstream_kw)
        except TypeError as e:
            print("Mock stream integrate failed: {}".format(str(e)))
            return None

        return stream

    def tracks_ln_likelihood(self, stream, galcen_frame):
        stream_c = stream.to_coord_frame(
            self.stream_frame, galactocentric_frame=galcen_frame)

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
        galcen_frame = self.get_galcen_frame(**pars['sun'])
        w0 = self.get_w0(galcen_frame, **pars['w0'])
        H = self.get_hamiltonian(**pars['potential'])

        orbit = self.get_orbit(H, w0)
        if orbit is None:
            return -np.inf

        if orbit.t[-1] < orbit.t[0]:
            orbit = orbit[::-1]

        stream = self.get_mockstream(H, orbit)
        if stream is None:
            return -np.inf

        return self.tracks_ln_likelihood(stream, galcen_frame)
