import unittest
import math

import numpy as np

from geodata.geodata.point import Point, get_bearing, get_distance, get_interpolated_point
from geodata.helper.helper import get_digits


class TestPointMethods(unittest.TestCase):
    def setUp(self):
        # See example here: http://www.movable-type.co.uk/scripts/latlong.html
        # Values vary slightly due to the use of different earth_radius values
        # https://geodesyapps.ga.gov.au/vincenty-direct
        self.lat_start = math.radians(53.320556)  # 53°19′14″N
        lon_start = math.radians(-1.729722)  # 001°43′47″W
        lat_end = math.radians(53.188432)  # 53°11′18″N
        lon_end = math.radians(0.133333)    # 000°08'00"E
        self.start_point = Point([lon_start, self.lat_start])
        self.end_point = Point([lon_end, lat_end])
        self.angle = math.radians(96.021667)  # 096°01′18″
        self.distance = 124_801  # meters
        self.point_radians = Point([lon_start, self.lat_start], coordinates_unit='radians')
        self.point_degrees = Point([math.degrees(lon_start), math.degrees(self.lat_start)], coordinates_unit='degrees')
        self.accuracy = 10

    def test_constructor(self):
        for illegal_argument, error in [
            # disallow empty points
            [None, TypeError],
            [[], IndexError],
            # disallow other list items than lists
            [1, TypeError],
            [1.0, TypeError],
            ["1", IndexError],
            # disallow lists as list items with not exactly two items
            [[1], IndexError],
            [[1, 2, 3], ValueError],
            # disallow more than one list item
            [[[1, 1], [2, 2]], TypeError]
        ]:
            self.assertRaises(error, Point, illegal_argument)
        self.assertEqual(Point, type(Point([0, 1])))

    def test_append(self):
        point = Point([0, 0])
        with self.assertWarns(UserWarning):
            point.append([1, 1])

    def test_set_x_lon(self):
        point = Point([0, 0])
        value = 1
        point.set_x_lon(value)
        self.assertEqual(value, point.x_lon)
        self.assertEqual(value, point[0])

    def test_set_y_lat(self):
        point = Point([0, 0])
        value = 1
        point.set_y_lat(value)
        self.assertEqual(value, point.y_lat)
        self.assertEqual(value, point[1])

    def test_set_geo_reference_system(self):
        point = Point([0, 0])
        self.assertRaises(ValueError, point.set_geo_reference_system, 'invalid_value')
        try:
            point.set_geo_reference_system("cartesian")
            self.assertEqual("cartesian", point.get_geo_reference_system())
            point.set_geo_reference_system("latlon")
            self.assertEqual("latlon", point.get_geo_reference_system())
        except Exception:
            self.fail("Unexpected exception when invoking set_geo_reference_system().")

    def test_add_vector_(self):
        self.start_point.add_vector_(self.distance, self.angle)
        accuracy = 4
        self.assertEqual(get_digits(self.start_point.x_lon, accuracy), get_digits(self.end_point.x_lon, accuracy))
        self.assertEqual(get_digits(self.start_point.y_lat, accuracy), get_digits(self.end_point.y_lat, accuracy))

    def test_add_vector(self):
        calculated_end_point = self.start_point.add_vector(self.distance, self.angle)
        # test that start point was not changed
        self.assertEqual(self.lat_start, self.start_point.y_lat)
        self.assertNotEqual(self.lat_start, calculated_end_point.y_lat)
        accuracy = 4
        # test that end point is correctly calculated
        self.assertAlmostEqual(get_digits(self.end_point.x_lon, accuracy), get_digits(calculated_end_point.x_lon,
                                                                                      accuracy))

    def test_conversion_geo_reference_systems(self):
        lat_start = math.radians(53)
        lon_start = math.radians(21)
        point = Point([lon_start, lat_start])
        point.to_cartesian_()
        point.to_latlon_()
        accuracy = 10
        # conversion between geo-reference systems yields the approximate same coordinates
        self.assertEqual(get_digits(lon_start, accuracy), get_digits(point.x_lon, accuracy))
        self.assertEqual(get_digits(lat_start, accuracy), get_digits(point.y_lat, accuracy))
        # walk along the equator for 100 meters
        lat_start = math.radians(0)
        lon_start = math.radians(0)
        distance = 100  # meters
        y_new = 0 + distance / 1000  # move 0.1 km east along equator
        point = Point([lon_start, lat_start])
        point.add_vector_(distance, 0)
        point.to_cartesian_()
        self.assertEqual(get_digits(lon_start, accuracy), get_digits(point.x_lon, accuracy))
        self.assertEqual(get_digits(y_new, accuracy), get_digits(point.y_lat, accuracy))

        # conversion without instant modification does not change the original points
        lat_start = math.radians(0)
        lon_start = math.radians(0)
        point = Point([lon_start, lat_start])
        point.to_cartesian()
        self.assertEqual(lon_start, point.x_lon)
        self.assertEqual(lat_start, point.y_lat)

    def test_get_bearing(self):
        # test bearing in cartesian plane
        point_a = Point([0, 0], 'cartesian').to_latlon()
        point_b = Point([1, 1], 'cartesian').to_latlon()
        # expected angle between point A and B is 45 degrees
        expected_bearing = math.radians(45)
        self.assertAlmostEqual(expected_bearing, get_bearing(point_a, point_b), places=4)
        # test bearing on Earth
        self.assertAlmostEqual(self.angle, get_bearing(self.start_point, self.end_point), places=3)
        # warning is thrown, when points do not have the same coordinates unit
        self.assertWarns(UserWarning, get_bearing, point_a, point_b.to_degrees())

    def test_get_distance(self):
        # test distance in cartesian plane
        point_a = Point([0, 0], 'cartesian')
        point_b = Point([1, 1], 'cartesian')
        # Pythagorean theorem: 1² + 1² = c² -> c = sqrt(c²)
        self.assertAlmostEqual(math.sqrt(2), get_distance(point_a, point_b))
        # test distance on Earth
        self.assertAlmostEqual(self.distance, get_distance(self.start_point, self.end_point), delta=1)
        end_point_degrees = self.end_point.to_degrees()
        self.assertAlmostEqual(self.distance, get_distance(self.start_point, end_point_degrees), delta=1)
        # warning is thrown, when points do not have the same coordinates unit
        self.assertWarns(UserWarning, get_distance, point_a.to_latlon(), point_b.to_latlon().to_degrees())

    def test_get_interpolated_point(self):
        ratio = 0.5
        # test interpolation on Earth
        interpolated_point = get_interpolated_point(self.start_point, self.end_point, ratio)
        self.assertAlmostEqual(get_bearing(self.start_point, self.end_point),
                               get_bearing(self.start_point, interpolated_point), places=4)

    def test_point_copy(self):
        point = Point([0, 0], geo_reference_system='cartesian')
        point_list = [point]
        point_copy = point.deep_copy()
        point_copy.set_x_lon(1)
        point_copy.to_latlon_()
        # changing the copy does not change the original point
        self.assertEqual(0, point.x_lon)
        self.assertEqual('cartesian', point.get_geo_reference_system())
        # the original point object is not changed
        self.assertEqual(point, point_list[0])

    def test_to_radians(self):
        point_radians = self.point_radians.deep_copy()
        point_degrees = self.point_degrees.deep_copy()
        # successfully converts between degrees and radians and changes the coordinates_unit
        point = point_degrees.deep_copy()
        self.assertAlmostEqual(point_radians.x_lon, point.to_radians().x_lon, places=self.accuracy)
        self.assertAlmostEqual(point_radians.y_lat, point.to_radians().y_lat, places=self.accuracy)
        self.assertEqual('radians', point_degrees.to_radians().get_coordinates_unit())
        # point is not modified
        self.assertEqual(point_degrees.x_lon, point.x_lon)

        # cannot convert into degrees if geo reference system is not 'latlon'
        point = point_radians.to_cartesian()
        self.assertRaises((ValueError, Exception), point.to_radians)

        # if already in radians, throws a warning and does not change the point coordinates
        with self.assertWarns(Warning):
            point = point_radians.to_radians()
        self.assertEqual(point_radians.x_lon, point.x_lon)

    def test_to_radians_(self):
        point_radians = self.point_radians.deep_copy()
        point_degrees = self.point_degrees.deep_copy()

        # successfully converts between degrees and radians and changes the coordinates_unit while point is modified
        point = point_degrees.deep_copy()
        point.to_radians_()
        self.assertAlmostEqual(point_radians.x_lon, point.x_lon, places=self.accuracy)
        self.assertAlmostEqual(point_radians.y_lat, point.y_lat, places=self.accuracy)
        self.assertEqual('radians', point.get_coordinates_unit())

        # cannot convert into degrees if geo reference system is not 'latlon'
        with self.assertRaises(ValueError):
            point_degrees.to_cartesian().to_radians_()

        # if already in radians, throws a warning and does not change the point coordinates
        with self.assertWarns(Warning):
            point_radians.deep_copy().to_radians_()

    def test_to_degrees(self):
        point_radians = self.point_radians.deep_copy()
        point_degrees = self.point_degrees.deep_copy()
        point = point_radians.deep_copy()

        # successfully converts between degrees and radians and changes the coordinates_unit
        self.assertAlmostEqual(point_degrees.x_lon, point_radians.to_degrees().x_lon, places=self.accuracy)
        self.assertAlmostEqual(point_degrees.y_lat, point_radians.to_degrees().y_lat, places=self.accuracy)
        self.assertEqual('degrees', point_radians.to_degrees().get_coordinates_unit())
        # point is not modified
        self.assertEqual(point_radians.x_lon, point.x_lon)

        # cannot convert into degrees if geo reference system is not 'latlon'
        point = point_radians.to_cartesian()
        self.assertRaises((ValueError, Exception), point.to_degrees)

        # if already in radians, throws a warning and does not change the point coordinates
        with self.assertWarns(Warning):
            point_degrees.to_degrees()

    def test_to_degrees_(self):
        point_radians = self.point_radians.deep_copy()
        point_degrees = self.point_degrees.deep_copy()

        # successfully converts between degrees and radians and changes the coordinates_unit while point is modified
        point = point_radians.deep_copy()
        point.to_degrees_()
        self.assertAlmostEqual(point_degrees.x_lon, point.x_lon, places=self.accuracy)
        self.assertAlmostEqual(point_degrees.y_lat, point.y_lat, places=self.accuracy)
        self.assertEqual('degrees', point.get_coordinates_unit())

        # cannot convert into degrees if geo reference system is not 'latlon'
        with self.assertRaises(ValueError):
            point_radians.to_cartesian().to_degrees_()

        # if already in degrees, throws a warning and does not change the point coordinates
        with self.assertWarns(Warning):
            point_degrees.deep_copy().to_degrees_()

    def test_is_coordinates_unit_valid(self):
        for illegal_argument, parameter in [
            [[np.pi + 0.1, np.pi], 'radians'],
            [[180.1, 90], 'degrees']
        ]:
            self.assertRaises(Exception, Point, illegal_argument, coordinates_unit=parameter)

        for valid_argument, parameter in [
            [[-180, -90], 'degrees'],
            [[180, 90], 'degrees'],
            [[-np.pi, -np.pi], 'radians'],
            [[np.pi, np.pi], 'radians']
        ]:
            try:
                Point(valid_argument, coordinates_unit=parameter)
            except Exception:
                self.fail("Unexpected exception when invoking set_geo_reference_system().")


if __name__ == "__main__":
    unittest.main()
