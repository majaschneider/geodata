"""Provides a point datatype for geo-coordinates and timestamps and their manipulation.
"""

from geodata.point import Point


class PointT(Point):
    """A point specifying a geographical location and a timestamp.
    """

    def __init__(self, coordinates, timestamp, geo_reference_system="latlon"):
        """
        Creates a new PointT object.

        Parameters
        ----------
        coordinates : list
            Contains the x- and y-coordinate of this point in the form [x,y]. If geo_reference_system
            is 'latlon', the values [x,y] refer to [longitude, latitude] in radian.
        timestamp
            The timestamp assigned to this point.
        geo_reference_system : {'latlon', 'cartesian'}
            Geographical reference system of the coordinates:
            - 'latlon': latitude and longitude coordinates on earth
            - 'cartesian': uses Euclidean space
        """
        super().__init__(coordinates, geo_reference_system)
        self.timestamp = timestamp
