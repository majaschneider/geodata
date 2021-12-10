import unittest
import math

from de4l_geodata.geodata.point import Point, get_bearing, get_distance, get_interpolated_point
from de4l_geodata.helper.helper import get_digits


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

    def test_constructor(self):
        with self.assertRaises(TypeError):
            # disallow empty points
            Point()
            Point([])
            # disallow other list items than lists
            Point(1)
            Point(1.0)
            Point("1")
            # disallow lists as list items with not exactly two items
            Point([1])
            Point([1, 2, 3])
            # disallow more than one list item
            Point([[1, 1], [2, 2]])
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
        with self.assertRaises(ValueError):
            point.set_geo_reference_system("invalid_value")
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
        self.assertAlmostEqual(expected_bearing, get_bearing(point_a, point_b))
        # test bearing on Earth
        self.assertAlmostEqual(self.angle, get_bearing(self.start_point, self.end_point), places=3)

    def test_get_distance(self):
        # test distance in cartesian plane
        point_a = Point([0, 0], 'cartesian')
        point_b = Point([1, 1], 'cartesian')
        # Pythagorean theorem: 1² + 1² = c² -> c = sqrt(c²)
        self.assertAlmostEqual(math.sqrt(2), get_distance(point_a, point_b))
        # test distance on Earth
        self.assertAlmostEqual(self.distance, get_distance(self.start_point, self.end_point), delta=1)

    def test_get_interpolated_point(self):
        ratio = 0.5
        # test interpolation on Earth
        interpolated_point = get_interpolated_point(self.start_point, self.end_point, ratio)
        self.assertAlmostEqual(get_bearing(self.start_point, self.end_point),
                               get_bearing(self.start_point, interpolated_point))

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


if __name__ == "__main__":
    unittest.main()
