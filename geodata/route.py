"""Provides a route datatype for point lists (geo-coordinates) and their manipulation."""

import torch
from geodata.point import Point


class Route(list):
    """A route indicates a sequence of points."""

    def __init__(self, route=None):
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
        if not isinstance(value, Point):
            value = Point(value)
        super().__setitem__(key, value)

    @classmethod
    def from_torch_tensor(cls, tensor):
        """
        Convert tensor from torch.Tensor to Route.
        :param tensor: route of type torch.Tensor
        :return: route of type Route
        """
        return cls(tensor.numpy().tolist())

    def append(self, value):
        if not isinstance(value, Point):
            value = Point(value)
        super().append(value)

    def scale(self, scale_values):
        """
        Scales route coordinates from minimum and maximum values indicated by scale_values parameter to [0,1].
        :param scale_values: minimum and maximum values to scale route points with, provided in format
        (x minimum, x maximum, y minimum, y maximum) for coordinates
        :return: route scaled by scale_values
        """
        x_min, x_max, y_min, y_max = scale_values
        for point in self:
            point.set_x_lon((point.x_lon - x_min) / (x_max - x_min))
            point.set_y_lat((point.y_lat - y_min) / (y_max - y_min))
        return self

    def inverse_scale(self, scale_values):
        """
        Scales route coordinates from [0,1] to minimum and maximum values indicated by scale_values parameter.
        :param scale_values: minimum and maximum values to scale route points to, provided in format
        (x minimum, x maximum, y minimum, y maximum) for coordinates
        :return: route scaled to scale_values
        """
        (x_min, x_max, y_min, y_max) = scale_values
        for point in self:
            point.set_x_lon(point.x_lon * (x_max - x_min) + x_min)
            point.set_y_lat(point.y_lat * (y_max - y_min) + y_min)
        return self

    def pad(self, target_len):
        """
        Pads route with zero values to achieve target_len.
        :param target_len: target length of route
        :return padded route
        """
        route = self
        pad_len = target_len - len(self)
        if pad_len > 0:
            tensor = torch.tensor(self)
            pad = torch.nn.ZeroPad2d((0, 0, 0, pad_len))
            tensor = pad(tensor)
            route = self.__init__(tensor.numpy().tolist())
        return route
