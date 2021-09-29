"""Provides a point datatype for geo-coordinates and their manipulation.
"""

import math
import warnings
import numpy as np


class Point(list):
    """A point specifying a geographical location.
    """

    def __init__(self, coordinates, geo_reference_system="latlon"):
        """
        Creates a new Point object.

        Parameters
        ----------
        coordinates : list
            Contains the x- and y-coordinate of this point in the form [x,y]. If geo_reference_system
            is 'latlon', the values [x,y] refer to [longitude, latitude] in radian.
        geo_reference_system : {'latlon', 'cartesian'}
            Geographical reference system of the coordinates:
            - 'latlon': latitude and longitude coordinates on earth
            - 'cartesian': uses Euclidean space
        """
        super().__init__(coordinates)
        self.__geo_reference_system = None
        self.set_geo_reference_system(geo_reference_system)
        self.__earth_radius = 6_378_137
        self.x_lon = coordinates[0]
        self.y_lat = coordinates[1]
        assert isinstance(coordinates, list)
        assert len(coordinates) == 2
        for i in range(2):
            assert type(coordinates[i]) in (int, float, np.float64)

    def append(self, obj):
        warnings.warn("Point class does not provide append functionality. Use set instead.")

    def __setitem__(self, key, value):
        """
        Sets the value of the coordinate indicated by key.

        Parameters
        ----------
        key : {0, 1}
            The coordinate key:
            - 0 for the x-coordinate respectively longitude
            - 1 for the y-coordinate respectively latitude
        value : float
            The coordinate value.
        """
        super().__setitem__(key, value)
        if key == 0:
            self.x_lon = value
        if key == 1:
            self.y_lat = value

    def set_x_lon(self, value):
        """
        Sets the x coordinate or longitude of this point.

        Parameters
        ----------
        value : float
            New x-coordinate respectively longitude of this point.
        """
        self.__setitem__(0, value)

    def set_y_lat(self, value):
        """
        Sets the y coordinate or latitude of this point.

        Parameters
        ----------
        value : float
            New y-coordinate respectively latitude of this point.
        """
        self.__setitem__(1, value)

    def set_geo_reference_system(self, value):
        """
        Sets the geo reference system that this point's coordinates refer to.

        Parameters
        ----------
        value : {'latlon', 'cartesian'}
            New geographical reference system of this point:
            - 'latlon': latitude and longitude coordinates on earth
            - 'cartesian': uses Euclidean space
        """
        assert value in ("cartesian", "latlon")
        self.__geo_reference_system = value

    def add_vector(self, distance, angle):
        """
        Adds a vector to a point. The vector is defined by its length and angle. For details see 'Destination point
        given distance and bearing from start point' at http://www.movable-type.co.uk/scripts/latlong.html

        Parameters
        ----------
        distance : float
            Vector length in meters.
        angle : float
            Angle of vector in radian.
        """
        if self.__geo_reference_system == "latlon":
            angular_distance = distance / self.__earth_radius
            latitude_tmp = math.asin(
                math.sin(self.y_lat) * math.cos(angular_distance)
                + math.cos(self.y_lat) * math.sin(angular_distance) * math.cos(angle)
            )
            longitude_tmp = self.x_lon + math.atan2(
                math.sin(angle) * math.sin(angular_distance) * math.cos(self.y_lat),
                math.cos(angular_distance) - math.sin(self.y_lat) * math.sin(latitude_tmp),
            )
            # normalize to [-180,180]
            longitude_tmp = (longitude_tmp + 3 * math.pi) % (2 * math.pi) - math.pi
            self.set_x_lon(longitude_tmp)
            self.set_y_lat(latitude_tmp)
        else:
            raise NotImplementedError("Adding a vector onto a cartesian point is not available.")

    def to_cartesian(self):
        """
        Transforms coordinates of this point from latitude and longitude (both in radian) into cartesian.
        """
        if self.__geo_reference_system == "latlon":
            r = self.__earth_radius / 1000  # km
            self.set_x_lon(r * self.x_lon)
            self.set_y_lat(r * np.log(np.tan(np.pi / 4.0 + self.y_lat / 2.0)))
            self.set_geo_reference_system("cartesian")
        else:
            warnings.warn("geo reference system is already cartesian.")

    def to_latlon(self):
        """
        Transforms coordinates of this point from cartesian into latitude and longitude (both in radian).
        """
        if self.__geo_reference_system == "cartesian":
            r = self.__earth_radius / 1000  # km
            self.set_x_lon(self.x_lon / r)
            self.set_y_lat(np.pi / 2 - 2 * np.arctan(np.exp(-self.y_lat / r)))
            self.set_geo_reference_system("latlon")
        else:
            warnings.warn("geo reference system is already latlon.")
