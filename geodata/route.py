"""Provides a route datatype for lists of points (geo-coordinates) and their manipulation.
"""

import torch
from geodata.point import Point


class Route(list):
    """A route indicating a sequence of points.
    """

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
            assert isinstance(route, list)
            # make sure list items are of type Point
            for idx, point in enumerate(route):
                if not isinstance(point, Point):
                    self.__setitem__(idx, Point(point))

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
        return cls(tensor.numpy().tolist())

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
        Pads route with zero values to achieve target_len.

        Parameters
        ----------
        target_len : int
            Target length of route.

        Returns
        -------
        Route
            A copy of this route, padded by zero values to target_length.
        """
        route = self
        pad_len = target_len - len(self)
        if pad_len > 0:
            tensor = torch.tensor(self)
            pad = torch.nn.ZeroPad2d((0, 0, 0, pad_len))
            tensor = pad(tensor)
            route = self.__init__(tensor.numpy().tolist())
        return route
