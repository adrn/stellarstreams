# Third-party
import astropy.coordinates as coord
import astropy.units as u
from gala.coordinates.greatcircle import (GreatCircleICRSFrame,
                                          greatcircle_transforms)

default_docstring = """A coordinate frame for the {stream_name} stream defined
in {reference}.

Parameters
----------
phi1 : `~astropy.units.Quantity`
    Longitude component.
phi2 : `~astropy.units.Quantity`
    Latitude component.
distance : `~astropy.units.Quantity`
    Distance.

pm_phi1_cosphi2 : `~astropy.units.Quantity`
    Proper motion in longitude.
pm_phi2 : `~astropy.units.Quantity`
    Proper motion in latitude.
radial_velocity : `~astropy.units.Quantity`
    Line-of-sight or radial velocity.
"""

def stream_coord_factory(cls_name, stream_name, reference):

    @greatcircle_transforms(self_transform=False)
    class StreamCls(GreatCircleICRSFrame):
        pole = coord.ICRS(ra=72.2643 * u.deg,
                          dec=-20.6575 * u.deg)
        ra0 = 160 * u.deg
        rotation = 0 * u.deg

    StreamCls.__name__ = cls_name
    StreamCls.__doc__ = default_docstring.format(stream_name=stream_name,
                                                 reference=reference)

    return StreamCls
