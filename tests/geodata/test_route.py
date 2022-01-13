import random
import unittest
import math

import torch
from pandas import Timestamp

from de4l_geodata.geodata.route import Route, degrees_to_radians
from de4l_geodata.geodata.point import Point
from de4l_geodata.geodata.point_t import PointT


class TestPointMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.route_without_timestamps = Route([Point([0, 0])])
        self.route_with_timestamps = Route([PointT([0, 0], timestamp=Timestamp(0))])

    def test_constructor(self):
        # initializes with empty list
        self.assertEqual(Route([]), Route())
        with self.assertRaises(TypeError):
            # disallow other list items than lists
            Route(1)
            Route(1.0)
            Route("1")
            # disallow lists as list items that are not of type Point or can be converted into Point
            Route([[]])
            Route([[1]])
            Route([[1, 2, 3]])
        # successfully creates a Route
        self.assertEqual(Route, type(Route([[0, 0]])))
        self.assertEqual(Route, type(Route([[0, 0], [1, 1]])))
        self.assertEqual(Route, type(Route([Point([0, 0])])))
        self.assertEqual(Route, type(Route([Point([0, 0]), Point([1, 1])])))
        # successfully creates a route with timestamps
        point_0 = [0, 0]
        point_1 = [1, 1]
        route = [point_0, point_1]
        timestamp_0 = Timestamp(year=2022, month=1, day=2, hour=6, minute=30, second=0)
        timestamp_1 = Timestamp(year=2022, month=1, day=2, hour=6, minute=30, second=5)
        timestamps = [timestamp_0, timestamp_1]
        expected_route = Route([PointT(point_0, timestamp=timestamp_0), PointT(point_1, timestamp=timestamp_1)])
        self.assertEqual(expected_route, Route(route, timestamps))

    def test_set_item(self):
        # assures that list items are points
        route = Route([[0, 0]])
        self.assertEqual(Point, type(route[0]))

    def test_from_torch_tensor(self):
        tensor = torch.tensor([[0., 0.], [1., 1.]], requires_grad=True)
        route = Route([[0, 0], [1, 1]])
        route_from_tensor = Route.from_torch_tensor(tensor)
        self.assertEqual(route, route_from_tensor)

    def test_append(self):
        route = Route()
        route.append([0, 0])
        self.assertEqual(Route([[0, 0]]), route)

    def test_scale(self):
        scale_values = (-1, 1, -1, 1)
        r = Route([[-1, 1]])
        r.scale(scale_values)
        self.assertEqual(0, r[0].x_lon)
        self.assertEqual(1, r[0].y_lat)

    def test_inverse_scale(self):
        scale_values = (-1, 1, -1, 1)
        route = Route([[-1, 1]])
        route.scale(scale_values)
        route.inverse_scale(scale_values)
        self.assertEqual(-1, route[0].x_lon)
        self.assertEqual(1, route[0].y_lat)

    def test_pad(self):
        # method only applicable for routes containing items of type Point, but no subclasses of Point
        with self.assertRaises(Exception):
            self.route_with_timestamps.pad(5)
        try:
            self.route_without_timestamps.pad(5)
        except Exception:
            self.fail("Unexpected exception when invoking pad().")

        route = Route([[1, 1]])
        original_len = len(route)
        target_len = 3
        route.pad(target_len)
        self.assertEqual(target_len, len(route))
        for point in route[original_len:]:
            self.assertEqual(Point, type(point))
            self.assertEqual(0, point.x_lon)
            self.assertEqual(0, point.y_lat)

    def test_sort_by_time(self):
        # method only applicable for routes containing items of type PointT, but not Point
        with self.assertRaises(Exception):
            self.route_without_timestamps.sort_by_time()
        try:
            self.route_with_timestamps.sort_by_time()
        except Exception:
            self.fail("Unexpected exception when invoking sort_by_time().")

        route = Route()
        for i in range(100):
            route.append(PointT([1, 1], timestamp=Timestamp(random.randint(0, 1_000))))
        route.sort_by_time()
        timestamps = [p.timestamp for p in route]
        self.assertTrue(timestamps == sorted(timestamps))

    def test_has_timestamps(self):
        self.assertFalse(self.route_without_timestamps.has_timestamps())
        self.assertTrue(self.route_with_timestamps.has_timestamps())

    def test_deep_copy(self):
        route_without_timestamps = Route([[0, 0], [1, 1]])
        route_with_timestamps = Route([PointT([0, 0], timestamp=Timestamp(0)),
                                       PointT([1, 1], timestamp=Timestamp(1))])
        self.assertTrue(route_with_timestamps.deep_copy().has_timestamps())
        for route in [route_without_timestamps, route_with_timestamps]:
            route_copy = route.deep_copy()
            route_copy[0].set_x_lon(5)
            route_copy[0].to_cartesian_()
            self.assertEqual(0, route[0].x_lon)
            self.assertEqual('latlon', route[0].get_geo_reference_system())

    def test_degrees_to_radians(self):
        coordinates_list_degrees = [[-8, 41], [-8, 41]]
        coordinates_list_radians = [[math.radians(-8), math.radians(41)], [math.radians(-8), math.radians(41)]]
        self.assertEqual(coordinates_list_radians, degrees_to_radians(coordinates_list_degrees))

    def test_delete_item_(self):
        route = Route([[0, 0], [1, 1]]).delete_point_at_(1)

        self.assertEqual(Route([[0, 0]]), route)
        self.assertEqual(Route, type(route))
        with self.assertRaises(KeyError):
            route.delete_point_at_(2)

        route_with_timestamps = Route([PointT([0, 0], Timestamp(0)), PointT([1, 1], Timestamp(1))])
        self.assertTrue(route_with_timestamps.delete_point_at_(1).has_timestamps())


if __name__ == "__main__":
    unittest.main()
