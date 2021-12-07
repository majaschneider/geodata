"""Provides a route datatype for lists of points (geo-coordinates) and their manipulation.
"""

import torch
from de4l_geodata.geodata.point import Point
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
        return len(self) > 0 and isinstance(self[0], PointT)

    def __init__(self, route=None):
        """
        Creates a new Route object.

        Parameters
        ----------
        route : list, optional
            The route, that this route should be initialized with.
        """
        # initialize with empty list
        super().__init__()
        if route is not None:
            # set list items if any
            super().__init__(route)
            if not isinstance(route, list):
                raise ValueError("If route is provided, it needs to be of type list.")
            # make sure list items are of type Point
            for idx, point in enumerate(route):
                if not isinstance(point, Point):
                    self.__setitem__(idx, Point(point))
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
        value : list
            The point that is to be appended to this route.

        Returns
        -------
        Route
            This route which is appended by value.
        """
        if not isinstance(value, Point):
            value = Point(value)
        super().append(value)
        if self.has_timestamps():
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
