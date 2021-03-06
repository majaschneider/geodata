import random
import unittest
import math

import pandas as pd
import numpy as np
import torch
from pandas import Timestamp, Timedelta

from geodata.geodata.route import Route
from geodata.geodata.point import Point, get_distance
from geodata.geodata.point_t import PointT


class TestPointMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.route_without_timestamps = Route([Point([0, 0])])
        self.route_with_timestamps = Route([PointT([0, 0], timestamp=Timestamp(0))])

    def test_init(self):
        # initializes with empty list
        self.assertEqual(Route([]), Route())

        for illegal_route_argument in [
            # disallow other list items than lists
            1.0,
            "1",
            # disallow lists as list items that are not of type Point or can be converted into Point
            [[]],
            [[1]],
            [[1, 2, 3]]
        ]:
            self.assertRaises(TypeError, Route.__init__, illegal_route_argument)

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

        # unit is derived from points that route is initialized with
        point_degrees = Point([-8, 41], coordinates_unit='degrees')
        point_radians = Point([0, 0], coordinates_unit='radians')
        self.assertEqual('radians', Route([[0, 0], point_radians]).get_coordinates_unit())
        self.assertEqual('radians', Route([point_radians]).get_coordinates_unit())
        self.assertEqual('degrees', Route([point_degrees]).get_coordinates_unit())
        self.assertEqual('degrees', Route([point_degrees, point_degrees]).get_coordinates_unit())
        # check that points have equal unit, otherwise raise error
        for illegal_route_argument in [
            [[0, 0], point_degrees],
            [point_radians, point_degrees]
        ]:
            self.assertRaises(Exception, Route, illegal_route_argument)

        # if no points but list provided at init, assume coordinates unit is point default 'radians'
        self.assertEqual('radians', Route([[0, 0], [1, 1]]).get_coordinates_unit())
        # if points list and coordinates_unit parameter provided, initializes the list with the given unit
        route = Route([[-8, 41], [-8.1, 41.1]], coordinates_unit='degrees')
        self.assertEqual('degrees', route.get_coordinates_unit())
        self.assertEqual([-8, 41], route[0])
        # if points of type Point and coordinates_unit parameter provided, throws an error if mismatch
        self.assertRaises(Exception, Route, [point_degrees], coordinates_unit='radians')
        self.assertRaises(Exception, Route, [point_radians], coordinates_unit='degrees')

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
        point_degrees = Point([-8, 41], coordinates_unit='degrees')
        point_radians = Point([3, 3], coordinates_unit='radians')
        point_with_timestamp = PointT([3, 3], timestamp=pd.Timestamp(1))

        # successfully appends lists to empty routes
        route = Route()
        route.append([2, 2])
        self.assertEqual(Route([[2, 2]]), route)

        # successfully appends points and has the correct coordinates_unit
        route = Route()
        route.append(point_degrees)
        self.assertEqual(Route([point_degrees]), route)
        self.assertEqual('degrees', route.get_coordinates_unit())

        route = Route()
        route.append(point_radians)
        self.assertEqual(Route([point_radians]), route)
        self.assertEqual('radians', route.get_coordinates_unit())

        # successfully appends lists to non-empty routes
        route_radians = Route([[2, 2]])
        route_radians.append([1, 1])
        self.assertEqual(Route([[2, 2], [1, 1]]), route_radians)
        self.assertEqual('radians', route_radians.get_coordinates_unit())

        route_radians = Route([point_radians])
        route_radians.append(point_radians)
        self.assertEqual(Route([point_radians, point_radians]), route_radians)
        self.assertEqual('radians', route_radians.get_coordinates_unit())

        route_degrees = Route([point_degrees])
        route_degrees.append(point_degrees)
        self.assertEqual(Route([point_degrees, point_degrees]), route_degrees)
        self.assertEqual('degrees', route_degrees.get_coordinates_unit())

        # successfully appends a point with timestamp to a route with timestamps
        route_with_timestamps = Route([point_with_timestamp])
        route_with_timestamps.append(point_with_timestamp)
        self.assertEqual(Route([[3, 3], [3, 3]]), route_with_timestamps)
        self.assertTrue(route_with_timestamps.has_timestamps())

        # when point with timestamp is added to route without timestamps, timestamp will be lost but point appended
        route_without_timestamps = Route([Point([3, 3])])
        with self.assertWarns(Warning):
            route_without_timestamps.append(point_with_timestamp)
        self.assertEqual(Route([[3, 3], [3, 3]]), route_without_timestamps)

        # appends a point with differing coordinates_unit and changes the point's coordinates_unit before appending
        route_radians = Route([[0.1, 0.5]])
        point_degrees = Point([-8, 41], coordinates_unit='degrees')
        point_radians = point_degrees.to_radians()
        with self.assertWarns(Warning):
            route_radians.append(point_degrees)
        self.assertEqual(Route([[0.1, 0.5], point_radians]), route_radians)

        route_degrees = Route([point_degrees])
        with self.assertWarns(Warning):
            route_degrees.append(point_radians)
        self.assertEqual(Route([point_degrees, point_degrees]), route_degrees)

        # prohibits appending a point with non-matching geo reference system
        route_latlon = Route([[1, 1], [2, 2]])
        point_cartesian = Point([3, 3], geo_reference_system='cartesian')
        self.assertRaises(Exception, route_latlon.append, point_cartesian)

        # checks if the point to be appended has valid coordinates
        with self.assertRaises(Exception):
            Route([[0.1, 0.5]]).append([np.pi + 0.1, np.pi])
        with self.assertRaises(Exception):
            Route([[-8, 41]], coordinates_unit='degrees').append([180.1, 90])

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

    def test_delete_item_(self):
        route = Route([[0, 0], [1, 1]]).delete_point_at_(1)

        self.assertEqual(Route([[0, 0]]), route)
        self.assertEqual(Route, type(route))
        with self.assertRaises(KeyError):
            route.delete_point_at_(2)

        route_with_timestamps = Route([PointT([0, 0], Timestamp(0)), PointT([1, 1], Timestamp(1))])
        self.assertTrue(route_with_timestamps.delete_point_at_(1).has_timestamps())

    def test_conversion_between_degrees_and_radians(self):
        # route copies are successfully changed
        route_degrees = Route([Point([-8, 41], coordinates_unit='degrees'),
                               Point([-8.1, 41.1], coordinates_unit='degrees')])
        route_radians = Route([Point([math.radians(-8), math.radians(41)]),
                               Point([math.radians(-8.1), math.radians(41.1)])])
        self.assertEqual(route_radians, route_degrees.to_radians())
        self.assertEqual(route_degrees, route_radians.to_degrees())

        # routes are modified instantly
        route = route_radians.deep_copy()
        route.to_degrees_()
        self.assertEqual(route_degrees, route)

        route = route_degrees.deep_copy()
        route.to_radians_()
        self.assertEqual(route_radians, route)

    def test_get_coordinates_unit(self):
        point_degrees_1 = Point([-8, 41], coordinates_unit='degrees')
        point_degrees_2 = Point([-8.1, 41.1], coordinates_unit='degrees')
        point_radians_1 = Point([math.radians(-8), math.radians(41)])
        point_radians_2 = Point([math.radians(-8.1), math.radians(41.1)])
        route_degrees = Route([point_degrees_1, point_degrees_2])
        route_radians = Route([point_radians_1, point_radians_2])
        self.assertEqual('radians', route_radians.get_coordinates_unit())
        self.assertEqual('degrees', route_degrees.get_coordinates_unit())
        invalid_points = [point_radians_1, point_degrees_1]
        self.assertRaises(Exception, Route, invalid_points)

    def test_get_geo_reference_system(self):
        point_cartesian = Point([1, 1], geo_reference_system='cartesian')
        point_latlon = Point([1, 1], coordinates_unit='radians', geo_reference_system='latlon')
        route_cartesian = Route([point_cartesian, point_cartesian])
        route_latlon = Route([point_latlon, point_latlon])
        self.assertEqual('cartesian', route_cartesian.get_geo_reference_system())
        self.assertEqual('latlon', route_latlon.get_geo_reference_system())
        invalid_points = [point_cartesian, point_latlon]
        self.assertRaises(Exception, Route, invalid_points)

    def test_max_speed(self):
        time_between_route_points = Timedelta(seconds=10)
        point_0 = Point([0, 0], 'cartesian')
        point_1 = Point([0, 100], 'cartesian')
        point_2 = Point([0, 50], 'cartesian')
        expected_max_speed_kmh = \
            get_distance(point_0, point_1) * 3_600 / (1_000 * time_between_route_points.total_seconds())
        route = Route([point_0, point_1, point_2])
        self.assertAlmostEqual(expected_max_speed_kmh, route.max_speed(time_between_route_points))

    def test_conversion_geo_reference_systems(self):
        route_cartesian = Route([Point([0, 0], 'cartesian'),
                                 Point([0, 100], 'cartesian'),
                                 Point([0, 150], 'cartesian')])
        route_converted = route_cartesian.to_latlon().to_cartesian()
        for i in range(len(route_cartesian)):
            self.assertAlmostEqual(
                Point(route_cartesian[i], 'cartesian').y_lat,
                Point(route_converted[i], 'cartesian').y_lat
            )

        route = route_cartesian.deep_copy()
        route.to_latlon_()
        for point in route:
            self.assertEqual('latlon', point.get_geo_reference_system())
        route.to_cartesian_()
        for point in route:
            self.assertEqual('cartesian', point.get_geo_reference_system())

        # coordinates unit is not changed even though the combination might not make sense
        route = Route([Point([0, 0], 'cartesian', 'degrees'),
                       Point([0, 100], 'cartesian', 'degrees'),
                       Point([0, 150], 'cartesian', 'degrees')])
        for point in route.to_latlon().to_degrees():
            self.assertEqual('degrees', point.get_coordinates_unit())
        route.to_latlon_()
        route.to_cartesian_()
        for point in route:
            self.assertEqual('degrees', point.get_coordinates_unit())

    def test_get_average_point(self):
        point_a = Point([0, 1], geo_reference_system='cartesian', coordinates_unit='radians')
        point_b = Point([3, 8], geo_reference_system='cartesian', coordinates_unit='radians')
        point_c = Point([6, -6], geo_reference_system='cartesian', coordinates_unit='radians')
        route = Route([point_a, point_b, point_c])
        expected_average_point = Point([3, 1], geo_reference_system='cartesian', coordinates_unit='radians')
        self.assertEqual(expected_average_point, route.get_average_point())
        self.assertIsNone(Route().get_average_point())


if __name__ == "__main__":
    unittest.main()
