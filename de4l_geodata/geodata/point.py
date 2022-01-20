"""Provides a point datatype for geo-coordinates and their manipulation.
"""

import math
import warnings
import numpy as np
import haversine as hs


def get_bearing(point_a, point_b):
    """
    Calculates the initial bearing between a start point A and an endpoint B. For details see 'Bearing' at
    http://www.movable-type.co.uk/scripts/latlong.html.

    Parameters
    ----------
    point_a : Point
        The start point in 'latlon' format and 'radians' unit.
    point_b : Point
        The endpoint in 'latlon' format and 'radians' unit.
    Returns
    -------
    bearing : float
        The initial bearing in radian, which followed in a straight line along a great-circle arc, starting at the
        start point will arrive at the end point.
    """
    if point_a.get_geo_reference_system() != 'latlon' or point_b.get_geo_reference_system() != 'latlon':
        raise ValueError("Both points need to be in 'latlon' format.")
    point_a = point_a.to_radians(ignore_warning=True)
    point_b = point_b.to_radians(ignore_warning=True)
    lon1, lat1 = point_a
    lon2, lat2 = point_b
    bearing = math.atan2(math.sin(lon2 - lon1) * math.cos(lat2),
                         math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
    return bearing


def get_distance(point_a, point_b):
    """
    Calculates the distance between two points.

    Parameters
    ----------
    point_a : Point
        The start point.
    point_b : Point
        The end point.

    Returns
    -------
    distance : float
        The distance between point_a and point_b in meters.
    """
    geo_ref_a = point_a.get_geo_reference_system()
    geo_ref_b = point_b.get_geo_reference_system()
    if geo_ref_a != geo_ref_b:
        raise ValueError("Both points need to have the same geo_reference_system.")
    if geo_ref_a == 'latlon':
        point_a = point_a.to_degrees(ignore_warning=True)
        point_b = point_b.to_degrees(ignore_warning=True)
        distance = hs.haversine([point_a.y_lat, point_a.x_lon], [point_b.y_lat, point_b.x_lon], hs.Unit.METERS)
    else:   # distance in cartesian plane
        distance = math.sqrt(math.pow(point_b.x_lon - point_a.x_lon, 2) + math.pow(point_b.y_lat - point_a.y_lat, 2))
    return distance


def get_interpolated_point(start_point, end_point, ratio):
    """
    Interpolates a point on the straight line between start point and end point, where the distance from the start
    point to the interpolated point corresponds to the provided ratio of the distance from the start point to the end
    point.

    Parameters
    ----------
    start_point : Point
        The start point of the line.
    end_point : Point
        The end point of the line.
    ratio : float
        The ratio of distance between start and interpolated to start and end point.

    Returns
    -------
    interpolated_point : Point
        The interpolated point with coordinates in the same unit as those of the start point.
    """
    geo_ref = start_point.get_geo_reference_system()
    if geo_ref == 'latlon':
        coordinates_unit = start_point.get_coordinates_unit()
        # calculate interpolation with radians unit coordinates
        start_point = start_point.to_radians(ignore_warning=True)
        interpolated_point = Point(start_point, geo_reference_system=geo_ref)
        interpolated_point.add_vector_(ratio * get_distance(start_point, end_point), get_bearing(start_point, end_point))
        # convert back to 'degrees' only if start point's coordinates unit was 'degrees'
        if coordinates_unit == 'degrees':
            interpolated_point.to_degrees_()
    else:
        raise NotImplementedError("Interpolating in the cartesian plane is not available.")
    return interpolated_point


class Point(list):
    """A point specifying a geographical location.
    """

    def __init__(self, coordinates, geo_reference_system="latlon", coordinates_unit="radians"):
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
        coordinates_unit : {'radians', 'degrees'}
            The coordinates unit of this point.
        """
        super().__init__(coordinates)
        self.__geo_reference_system = None
        self.set_geo_reference_system(geo_reference_system)
        self.__coordinates_unit = None
        self.set_coordinates_unit(coordinates_unit)
        self.__earth_radius = 6_371_000
        self.x_lon = coordinates[0]
        self.y_lat = coordinates[1]
        if not (isinstance(coordinates, list) and len(coordinates) == 2):
            raise ValueError("Coordinates need to be a list with two elements.")
        for i in range(2):
            if type(coordinates[i]) not in (int, float, np.float64):
                raise ValueError("Coordinates need to be of type int or float.")

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
        Returns
        -------
        point
            The modified point instance.
        """
        super().__setitem__(key, value)
        if key == 0:
            self.x_lon = value
        if key == 1:
            self.y_lat = value
        return self

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
        if value not in ("cartesian", "latlon"):
            raise ValueError("Geo reference system can only be 'latlon' or 'cartesian'.")
        self.__geo_reference_system = value

    def get_geo_reference_system(self):
        """
        Returns the geographical reference system of this point.

        Returns
        -------
        __geo_reference_system : {'latlon', 'cartesian'}
            The geographical reference system of this point.
        """
        return self.__geo_reference_system

    def to_degrees_(self, ignore_warning=False):
        """
        Converts the coordinates of this point into degrees unit, if the unit is 'radians' and the geo_reference_system
        is 'latlon'. If the geo_reference_system is 'cartesian', an error is thrown and no changes are made.

        Parameters
        ----------
        ignore_warning : bool
            If True, no warning will be thrown in case the point's coordinates unit is already 'degrees'.
        """
        if self.get_geo_reference_system() != 'latlon':
            raise ValueError("The coordinates can only be converted if the geo reference system is 'latlon.")
        if self.get_coordinates_unit() == 'degrees':
            if not ignore_warning:
                warnings.warn("Coordinates unit is already 'degrees'.")
        else:
            self.set_x_lon(math.degrees(self.x_lon))
            self.set_y_lat(math.degrees(self.y_lat))
            self.set_coordinates_unit('degrees')

    def to_degrees(self, ignore_warning=False):
        """
        Returns a copy of this point with the coordinates changed into degrees unit, if the unit is 'radians' and the
        geo_reference_system is 'latlon'. If the geo_reference_system is 'cartesian', an error is thrown.

        Parameters
        ----------
        ignore_warning : bool
            If True, no warning will be thrown in case the point's coordinates unit is already 'degrees'.

        Returns
        -------
        Point
            A copy of this point where the coordinates have been converted into 'degrees' if the unit is 'radians' and
            the geo_reference_system is 'latlon'.
        """
        point_copy = self.deep_copy()
        point_copy.to_degrees_(ignore_warning)
        return point_copy

    def to_radians_(self, ignore_warning=False):
        """
        Converts the coordinates of this point into radians unit, if the unit is 'degrees' and the geo_reference_system
        is 'latlon'. If the geo_reference_system is 'cartesian', an error is thrown and no changes are made.

        Parameters
        ----------
        ignore_warning : bool
            If True, no warning will be thrown in case the point's coordinates unit is already 'radians'.
        """
        if self.get_geo_reference_system() != 'latlon':
            raise ValueError("The coordinates can only be converted if the geo reference system is 'latlon.")
        if self.get_coordinates_unit() == 'radians':
            if not ignore_warning:
                warnings.warn("Coordinates unit is already 'radians'.")
        else:
            self.set_x_lon(math.radians(self.x_lon))
            self.set_y_lat(math.radians(self.y_lat))
            self.set_coordinates_unit('radians')

    def to_radians(self, ignore_warning=False):
        """
        Returns a copy of this point with the coordinates changed into radians unit, if the unit is 'degrees' and the
        geo_reference_system is 'latlon'. If the geo_reference_system is 'cartesian', an error is thrown.

        Parameters
        ----------
        ignore_warning : bool
            If True, no warning will be thrown in case the point's coordinates unit is already 'degrees'.

        Returns
        -------
        Point
            A copy of this point where the coordinates have been converted into 'radians' if the unit is 'degrees' and
            the geo_reference_system is 'latlon'.
        """
        point_copy = self.deep_copy()
        point_copy.to_radians_(ignore_warning)
        return point_copy

    def set_coordinates_unit(self, value):
        """
        Sets the value of __coordinates_unit.

        Parameters
        ----------
        value : {'radians', 'degrees'}
            The unit of this point's coordinates if their geo reference system is 'latlon'.
        """
        if value not in ("radians", "degrees"):
            raise ValueError("Coordinates unit can only be 'radians' or 'degrees'.")
        self.__coordinates_unit = value

    def get_coordinates_unit(self):
        """
        Returns the unit that this point's coordinates are in if the geo_reference_system is 'latlon'.

        Returns
        -------
        __coordinates_unit
            The unit of this point's coordinates if their geo_reference_system is 'latlon'.
        """
        return self.__coordinates_unit

    def add_vector_(self, distance, angle):
        """
        Adds a vector to this point and modifies it instantly. The vector is defined by its length and angle. For
        details see 'Destination point given distance and bearing from start point' at
        http://www.movable-type.co.uk/scripts/latlong.html

        Parameters
        ----------
        distance : float
            Vector length in meters.
        angle : float
            Angle of vector in radian.
        """
        if self.get_geo_reference_system() == "latlon":
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

    def add_vector(self, distance, angle):
        """
        Calculates the resulting point when a vector is added to this point. The vector is defined by its length and
        angle. For details see 'Destination point given distance and bearing from start point' at
        http://www.movable-type.co.uk/scripts/latlong.html

        Parameters
        ----------
        distance : float
            Vector length in meters.
        angle : float
            Angle of vector in radian.

        Returns
        -------
        Point
            A new point resulting from the input point to which the vector has been added.
        """
        point_copy = self.deep_copy()
        point_copy.add_vector_(distance, angle)
        return point_copy

    def to_cartesian_(self):
        """
        Transforms coordinates of this point from latitude and longitude (both in radian) into cartesian. The point is
        modified instantly.
        """
        if self.get_geo_reference_system() == "latlon":
            radius = self.__earth_radius / 1000  # km
            self.set_x_lon(radius * self.x_lon)
            self.set_y_lat(radius * np.log(np.tan(np.pi / 4.0 + self.y_lat / 2.0)))
            self.set_geo_reference_system("cartesian")
        else:
            warnings.warn("Geo reference system is already cartesian.")

    def to_cartesian(self):
        """
        Returns a copy of this point with coordinates changed from latitude and longitude (both in radian) into
        cartesian.

        Returns
        -------
        Point
            A copy of this point with coordinates transformed into cartesian format.
        """
        point_copy = self.deep_copy()
        point_copy.to_cartesian_()
        return point_copy

    def to_latlon_(self):
        """
        Transforms coordinates of this point from cartesian into latitude and longitude (both in radians). The point is
        modified instantly.
        """
        if self.get_geo_reference_system() == "cartesian":
            r = self.__earth_radius / 1000  # km
            self.set_x_lon(self.x_lon / r)
            self.set_y_lat(np.pi / 2 - 2 * np.arctan(np.exp(-self.y_lat / r)))
            self.set_geo_reference_system("latlon")
        else:
            warnings.warn("Geo reference system is already latlon.")
        return self

    def to_latlon(self):
        """
        Returns a copy of this point with coordinates changed from cartesian into latitude and longitude (both in
        radians).

        Returns
        -------
        Point
            A copy of this point with coordinates transformed into 'latlon' format.
        """
        point_copy = self.deep_copy()
        point_copy.to_latlon_()
        return point_copy

    def deep_copy(self):
        """
        Creates a deep copy of this point preserving its properties.

        Returns
        -------
        Point
            A deep copy of this point.
        """
        return Point(self, geo_reference_system=self.__geo_reference_system, coordinates_unit=self.__coordinates_unit)
