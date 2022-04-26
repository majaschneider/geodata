import unittest

import pandas as pd

from geodata.helper.helper import get_digits
from geodata.helper import parser
from geodata.geodata.route import Route


class TestHelper(unittest.TestCase):
    def test_get_digits(self):
        value = 0.123456789
        self.assertEqual(0, get_digits(value, 0))
        self.assertEqual(1, get_digits(value, 1))
        self.assertEqual(12345, get_digits(value, 5))
        self.assertEqual(123456789, get_digits(value, 9))
        self.assertEqual(1234567890, get_digits(value, 10))

    def test_route_str_to_list(self):
        for route_list in [
            [[-8.58, 41.14], [-8.5, 41.1]],
            []
        ]:
            self.assertEqual(route_list, parser.route_str_to_list(str(route_list)))

    def test_timestamps_str_to_list(self):
        for timestamps_list in [
            [pd.Timestamp('2020-01-01 10:00:00'), pd.Timestamp('2020-01-02 15:00:00')],
            []
        ]:
            self.assertEqual(timestamps_list, parser.timestamps_str_to_list(str(timestamps_list)))

    def test_float_str_to_list(self):
        for float_list in [
            [1, 2.5, 0, 7],
            []
        ]:
            self.assertEqual(float_list, parser.float_str_to_list(str(float_list)))

    def test_routes_str_to_list(self):
        route_degrees = Route([[-8.58, 41.14], [-8.5, 41.1]], coordinates_unit='degrees')
        route_radians = Route([[2, 2], [1, 1]])

        for routes_list in [
            [route_radians, route_radians],
            []
        ]:
            self.assertEqual(routes_list, parser.routes_str_to_list(str(routes_list)))

        route_list = [route_degrees, route_degrees]
        route_str = str(route_list)
        self.assertEqual(route_list, parser.routes_str_to_list(route_str, coordinates_unit='degrees'))
        with self.assertRaises(Exception):
            parser.routes_str_to_list(route_str, coordinates_unit='radians')

        route_list = [route_radians, route_degrees]
        route_str = str(route_list)
        with self.assertRaises(Exception):
            parser.routes_str_to_list(route_str, coordinates_unit='radians')
