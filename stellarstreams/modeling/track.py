# Third-party
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import binned_statistic

def get_stream_track(stream_c,
                     phi1_lim=[-180, 180]*u.deg,
                     phi1_binsize=1*u.deg):
    """TODO: document this shit

    Parameters
    ----------
    stream_c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
    phi1_lim : `~astropy.units.Quantity` (optional)
    phi1_binsize : `~astropy.units.Quantity` (optional)

    Returns
    -------
    mean_tracks : ``dict``
    std_tracks : ``dict``
    """
    phi1 = stream_c.spherical.lon.wrap_at(180*u.deg).degree
    phi1_lim = phi1_lim.to_value(u.deg)
    phi1_binsize = phi1_binsize.to_value(u.deg)

    phi1_bins = np.arange(phi1_lim[0], phi1_lim[1]+1e-8, phi1_binsize)
    phi1_binc = 0.5 * (phi1_bins[:-1] + phi1_bins[1:])

    means = dict()
    stds = dict()
    mean_tracks = dict()
    std_tracks = dict()
    units = dict()

    k = 'phi2'
    val = stream_c.spherical.lat
    means[k] = binned_statistic(phi1, val.value,
                                bins=phi1_bins, statistic='mean')
    stds[k] = binned_statistic(phi1, val.value,
                               bins=phi1_bins, statistic='std')
    units[k] = val.unit
    mean_tracks[k] = InterpolatedUnivariateSpline(phi1_binc,
                                                  means[k].statistic,
                                                  w=1/stds[k].statistic**2)
    std_tracks[k] = InterpolatedUnivariateSpline(phi1_binc,
                                                 stds[k].statistic)

    k = 'distmod'
    val = getattr(stream_c, 'distance').distmod
    means[k] = binned_statistic(phi1, val.value,
                                bins=phi1_bins, statistic='mean')
    stds[k] = binned_statistic(phi1, val.value,
                               bins=phi1_bins, statistic='std')
    units[k] = val.unit
    mean_tracks[k] = InterpolatedUnivariateSpline(phi1_binc, means[k].statistic,
                                                  w=1/stds[k].statistic**2)
    std_tracks[k] = InterpolatedUnivariateSpline(phi1_binc,
                                                 stds[k].statistic)

    has_vel = 's' in stream_c.data.differentials
    if has_vel:
        for k, k2 in [('pm1', 'd_lon_coslat'),
                      ('pm2', 'd_lat')]:
            val = getattr(stream_c.sphericalcoslat, k2)
            means[k] = binned_statistic(phi1, val.value,
                                        bins=phi1_bins, statistic='mean')
            stds[k] = binned_statistic(phi1, val.value,
                                       bins=phi1_bins, statistic='std')
            units[k] = val.unit
            mean_tracks[k] = InterpolatedUnivariateSpline(
                phi1_binc, means[k].statistic, w=1/stds[k].statistic**2)

            std_tracks[k] = InterpolatedUnivariateSpline(phi1_binc,
                                                         stds[k].statistic)

        k = 'rv'
        val = getattr(stream_c, 'radial_velocity')
        means[k] = binned_statistic(phi1, val.value,
                                    bins=phi1_bins, statistic='mean')
        stds[k] = binned_statistic(phi1, val.value,
                                   bins=phi1_bins, statistic='std')
        units[k] = val.unit
        mean_tracks[k] = InterpolatedUnivariateSpline(phi1_binc,
                                                      means[k].statistic,
                                                      w=1/stds[k].statistic**2)
        std_tracks[k] = InterpolatedUnivariateSpline(phi1_binc,
                                                     stds[k].statistic)

    return mean_tracks, std_tracks
