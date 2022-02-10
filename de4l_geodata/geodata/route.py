"""Provides a route datatype for lists of points (geo-coordinates) and their manipulation.
"""
import warnings

import torch
from de4l_geodata.geodata.point import Point, get_distance
from de4l_geodata.geodata.point_t import PointT


class Route(list):
    """A route indicating a sequence of points. If timestamps are given for each point, the route is sorted by time.
    """

    def has_timestamps(self):
        """
        Returns True, if the route points have a timestamp.

        Returns
        -------
        bool
            True, if route is not empty and points have timestamps, else False.
        """
        has_timestamps = False
        if len(self) > 0:
            has_timestamps = True
            for point in self:
                if not isinstance(point, PointT):
                    has_timestamps = False
        return has_timestamps

    def get_coordinates_unit(self):
        default_coordinates_unit = Point([0, 0]).get_coordinates_unit()
        if len(self) > 0:
            coordinates_unit = self[0].get_coordinates_unit()
            for point in self:
                if point.get_coordinates_unit() != coordinates_unit:
                    raise Exception("Not all points of route have the same coordinates unit.")
            return coordinates_unit
        else:
            return default_coordinates_unit

    def __init__(self, route=None, timestamps=None, coordinates_unit='radians'):
        """
        Creates a new Route object.

        Parameters
        ----------
        route : list, optional
            The route, that this route should be initialized with.
        timestamps : List, optional
            The list of timestamps of each route point.
        coordinates_unit : {'radians', 'degrees'}
            The coordinates unit of the route's points.
        """
        # initialize with empty list
        super().__init__()
        if route is not None:
            # set list items if any
            super().__init__(route)
            if not isinstance(route, list):
                raise ValueError(f"The provided route is not of type list but {type(route)}: {route}")
            if timestamps is not None:
                # make sure timestamps are of same length as route
                if not len(timestamps) == len(route):
                    raise ValueError("Timestamps and route need to be of same length.")
            # make sure list items are of type Point
            for idx, point in enumerate(route):
                if timestamps is not None:
                    self.__setitem__(idx, PointT(point, timestamps[idx], coordinates_unit=coordinates_unit))
                elif not isinstance(point, Point):
                    self.__setitem__(idx, Point(point, coordinates_unit=coordinates_unit))

            if self.has_timestamps():
                self.sort_by_time()

    def __setitem__(self, key, value):
        """
        Sets the value of the point at position key.

        Parameters
        ----------
        key : int
            Position at which value should be set.
        value : Point
            The new value.

        Returns
        -------
        Route
            The modified route instance.
        """
        if not isinstance(value, Point):
            value = Point(value)
        super().__setitem__(key, value)
        if self.has_timestamps():
            self.sort_by_time()
        return self

    @classmethod
    def from_torch_tensor(cls, tensor):
        """
        Create a Route object from a route in torch.Tensor format.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor object which is to be transformed into a Route object.

        Returns
        -------
        Route
            The tensor object transformed into a Route object.
        """
        return cls(tensor.detach().numpy().tolist())

    def append(self, value):
        """
        Appends a point to this route.

        Parameters
        ----------
        value : list or Point
            The point that is to be appended to this route. If it is in list format, it is converted into a Point object
            with default geo_reference_system ('latlon') and with the coordinates_unit of this route. If a point with
            timestamp is appended to a route that has no timestamps, the point is appended but the timestamp will be
            lost. On the other hand a point without timestamps cannot be appended to a route with timestamps.

        Returns
        -------
        Route
            This route which is appended by value.
        """
        route_has_timestamps = self.has_timestamps()
        if route_has_timestamps and not isinstance(value, PointT):
            raise Exception('Cannot append a point without a timestamp to a route that has timestamps.')
        if not isinstance(value, Point):
            value = Point(value, coordinates_unit=self.get_coordinates_unit())
        if not route_has_timestamps and isinstance(value, PointT):
            warnings.warn('A point with timestamp was added onto a route without timestamps. The point will be appended'
                          ' but the timestamp is removed.')
        if isinstance(value, Point) and value.get_coordinates_unit() != self.get_coordinates_unit():
            warnings.warn(f'Point had differing coordinates_unit than the route it was to be appended to. The point was'
                          f' converted to {self.get_coordinates_unit()} before appending.')
            if self.get_coordinates_unit() == 'degrees':
                value = value.to_degrees()
            else:
                value = value.to_radians()
        super().append(value)
        if route_has_timestamps:
            self.sort_by_time()
        return self

    def scale(self, scale_values):
        """
        Scales route coordinates from minimum and maximum values indicated by scale_values parameter to [0,1].

        Parameters
        ----------
        scale_values : tuple
            Minimum and maximum values to scale route points with, provided in format
            (x minimum, x maximum, y minimum, y maximum) for coordinates x and y.

        Returns
        -------
        Route
            This route scaled by scale_values.
        """
        x_min, x_max, y_min, y_max = scale_values
        for point in self:
            point.set_x_lon((point.x_lon - x_min) / (x_max - x_min))
            point.set_y_lat((point.y_lat - y_min) / (y_max - y_min))
        return self

    def inverse_scale(self, scale_values):
        """
        Scales route coordinates from [0,1] to minimum and maximum values indicated by scale_values parameter.

        Parameters
        ----------
        scale_values : tuple
            Minimum and maximum values to scale route points to, provided in format
            (x minimum, x maximum, y minimum, y maximum) for coordinates x and y.

        Returns
        -------
        Route
            This route scaled to scale_values.
        """
        (x_min, x_max, y_min, y_max) = scale_values
        for point in self:
            point.set_x_lon(point.x_lon * (x_max - x_min) + x_min)
            point.set_y_lat(point.y_lat * (y_max - y_min) + y_min)
        return self

    def pad(self, target_len):
        """
        Pads route with zero values to achieve target_len. Padding only applies to routes with items of type Point, but
        not of type being a subclass of Point.

        Parameters
        ----------
        target_len : int
            Target length of route.

        Returns
        -------
        Route
            A copy of this route, padded by zero values to target_length.
        """
        if len(self) > 0:
            if not type(self[0]) is Point:
                raise Exception("pad only applies to routes with items of type Point. No subclasses of Point are "
                                "allowed.")
        pad_len = target_len - len(self)
        if pad_len > 0:
            tensor = torch.tensor(self)
            pad = torch.nn.ZeroPad2d((0, 0, 0, pad_len))
            tensor = pad(tensor)
            self.__init__(tensor.numpy().tolist())
        return self

    def sort_by_time(self):
        """
        Sorts the items of this route by timestamp. This method only applies to routes with items of type PointT.

        Returns
        -------
        Route
            This route sorted by the timestamp of its items.

        """
        if len(self) > 0:
            if not isinstance(self[0], PointT):
                raise Exception("sort_by_time only applies to routes with items of type PointT.")
        self.sort(key=lambda item: item.timestamp)
        return self

    def deep_copy(self):
        """
        Creates a deep copy of this route preserving its properties.

        Returns
        -------
        Route
            A deep copy of this route.
        """
        route_copy = Route()
        for point in self:
            route_copy.append(point.deep_copy())
        return route_copy

    def get_timestamps(self):
        """
        Returns the timestamps of the route points as a list, if the route has timestamps.

        Returns
        -------
        timestamps : List
            The timestamps of the route points as a list or None if the route points have no timestamps.
        """
        timestamps = None
        if self.has_timestamps():
            timestamps = []
            for point in self:
                timestamps.append(point.timestamp)
        return timestamps

    def delete_point_at_(self, idx):
        """
        Removes point at position idx from this route. The method modifies this route instantly.

        Parameters
        ----------
        idx : int
            The index of the point to remove from this list.

        Returns
        -------
        Route
            This route without item at position idx.
        """
        if len(self) > 0 and len(self) > idx:
            self.__delitem__(idx)
        else:
            raise KeyError("idx is not valid. The route contains" + str(len(self)) + "points.")
        return self

    def to_cartesian(self, ignore_warnings=False):
        """
        Returns a copy of this route with each point of the route converted from a 'latlon' to a 'cartesian' geo
        reference system.

        Returns
        -------
        route_cartesian : Route
            A copy of this route with each point in a cartesian geo reference system.
        """
        route_copy = self.deep_copy()
        route_copy.to_cartesian_(ignore_warnings)
        return route_copy

    def to_cartesian_(self, ignore_warnings=False):
        """Converts each point of this route instantly from a 'latlon' to a 'cartesian' geo reference system.
        """
        for point in self:
            point.to_cartesian_(ignore_warnings)

    def to_latlon(self, ignore_warnings=False):
        """
        Returns a copy of this route with each point of the route converted from a 'cartesian' to a 'latlon' geo
        reference system.

        Returns
        -------
        route_cartesian : Route
            A copy of this route with each point in a latlon geo reference system.
        """
        route_copy = self.deep_copy()
        route_copy.to_latlon_(ignore_warnings)
        return route_copy

    def to_latlon_(self, ignore_warnings=False):
        """Converts each point of this route instantly from a 'cartesian' to a 'latlon' geo reference system.
        """
        for point in self:
            point.to_latlon_(ignore_warnings)

    def to_radians_(self):
        """
        Converts the coordinates of this route's points into radians unit, if their unit is 'degrees' and the
        geo_reference_system is 'latlon'. If the geo_reference_system is 'cartesian', an error is thrown and no changes
        are made.
        """
        for point in self:
            point.to_radians_()

    def to_radians(self):
        """
        Returns a copy of this route with the coordinates changed into radians unit, if the unit is 'degrees' and the
        geo_reference_system of its points is 'latlon'. If the geo_reference_system is 'cartesian', an error is thrown.

        Returns
        -------
        Route
            A copy of this route where the coordinates have been converted into 'radians' if the unit is 'degrees' and
            the geo_reference_system of its points is 'latlon'.
        """
        route_copy = self.deep_copy()
        route_copy.to_radians_()
        return route_copy

    def to_degrees_(self):
        """
        Converts the coordinates of this route's points into degrees unit, if their unit is 'radians' and the
        geo_reference_system is 'latlon'. If the geo_reference_system is 'cartesian', an error is thrown and no changes
        are made.
        """
        for point in self:
            point.to_degrees_()

    def to_degrees(self):
        """
        Returns a copy of this route with the coordinates changed into degrees unit, if the unit is 'radians' and the
        geo_reference_system of its points is 'latlon'. If the geo_reference_system is 'cartesian', an error is thrown.

        Returns
        -------
        Route
            A copy of this route where the coordinates have been converted into 'degrees' if the unit is 'radians' and
            the geo_reference_system of its points is 'latlon'.
        """
        route_copy = self.deep_copy()
        route_copy.to_degrees_()
        return route_copy

    def max_speed(self, time_between_route_points):
        """
        Returns the maximum speed in kilometers per hour of the taxi when driving this route, assuming that the time
        between consecutive route points is fixed to the indicated value.

        Parameters
        ----------
        time_between_route_points : pd.Timedelta
            The time between consecutive route points.

        Returns
        -------
        maximum_speed_kmh : float
            The maximum speed of the taxi in kilometers per hour, when driving the route.
        """
        maximum_speed_kmh = 0
        for i in range(len(self) - 1):
            distance = get_distance(self[i], self[i + 1])
            current_speed_ms = distance / time_between_route_points.total_seconds()
            current_speed_kmh = current_speed_ms * 3_600 / 1_000
            if current_speed_kmh > maximum_speed_kmh:
                maximum_speed_kmh = current_speed_kmh
        return maximum_speed_kmh
