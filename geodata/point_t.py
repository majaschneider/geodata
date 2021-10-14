"""Provides a point datatype for geo-coordinates and timestamps and their manipulation.
"""

from geodata.point import Point, get_interpolated_point as get_interpolated


def get_interpolated_point(start_point, end_point, ratio):
    """
    Interpolates a point on the straight line between start point and end point, where the distance from the start
    point to the interpolated point corresponds to the provided ratio of the distance from the start point to the end
    point. The timestamp of the interpolated point is set to the start_point's timestamp.

    Parameters
    ----------
    start_point : PointT
        The start point of the line.
    end_point : PointT
        The end point of the line.
    ratio : float
        The ratio of distance between start and interpolated to start and end point.

    Returns
    -------
    interpolated_point : PointT
        The interpolated point.
    """
    point = get_interpolated(start_point, end_point, ratio)
    interpolated_point_t = PointT(point, timestamp=start_point.timestamp,
                                  geo_reference_system=point.get_geo_reference_system())
    return interpolated_point_t


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
