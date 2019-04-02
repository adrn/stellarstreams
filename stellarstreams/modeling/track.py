# Third-party
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import binned_statistic

def get_stream_track(stream_c,
                     phi1_lim=[-180, 180]*u.deg,
                     phi1_binsize=1*u.deg,
                     units=None):
    """TODO: document this shit

    Parameters
    ----------
    stream_c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
    phi1_lim : `~astropy.units.Quantity` (optional)
    phi1_binsize : `~astropy.units.Quantity` (optional)
    units : `dict`

    Returns
    -------
    mean_tracks : ``dict``
    std_tracks : ``dict``
    """
    # All position and velocity component names:
    component_names = (
        list(stream_c.get_representation_component_names().keys()) +
        list(stream_c.get_representation_component_names('s').keys()))

    # If no units are provided:
    if units is None:
        units = dict()

    units['phi1'] = units.get('phi1',
                              getattr(stream_c, component_names[0]).unit)

    phi1 = stream_c.spherical.lon.wrap_at(180*u.deg).to_value(units['phi1'])
    phi1_lim = phi1_lim.to_value(units['phi1'])
    phi1_binsize = phi1_binsize.to_value(units['phi1'])

    phi1_bins = np.arange(phi1_lim[0], phi1_lim[1]+1e-8, phi1_binsize)
    phi1_binc = 0.5 * (phi1_bins[:-1] + phi1_bins[1:])

    means = dict()
    stds = dict()
    mean_tracks = dict()
    std_tracks = dict()

    for k in component_names[1:]:
        val = getattr(stream_c, k)
        if k in units:
            val = val.to_value(units[k])
        else:
            units[k] = val.unit
            val = val.value

        means[k] = binned_statistic(phi1, val,
                                    bins=phi1_bins, statistic='mean')
        stds[k] = binned_statistic(phi1, val,
                                   bins=phi1_bins, statistic='std')

        mask = np.isfinite(means[k].statistic)
        mean_tracks[k] = InterpolatedUnivariateSpline(phi1_binc[mask],
                                                      means[k].statistic[mask])
        mask = np.isfinite(stds[k].statistic)
        std_tracks[k] = InterpolatedUnivariateSpline(phi1_binc[mask],
                                                     stds[k].statistic[mask])

    return mean_tracks, std_tracks


def get_orbit_track(orbit_c,
                    phi1_lim=[-180, 180]*u.deg,
                    units=None):
    """TODO: document this shit

    Parameters
    ----------
    stream_c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
    phi1_lim : `~astropy.units.Quantity` (optional)
    phi1_binsize : `~astropy.units.Quantity` (optional)
    units : `dict`

    Returns
    -------
    mean_tracks : ``dict``
    std_tracks : ``dict``
    """
