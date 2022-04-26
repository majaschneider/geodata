import unittest
import math
import datetime

import pandas

from geodata.geodata.point_t import PointT, get_interpolated_point


class TestPointMethods(unittest.TestCase):
    def setUp(self):
        # See example here: http://www.movable-type.co.uk/scripts/latlong.html
        # Values vary slightly due to the use of different earth_radius values
        # https://geodesyapps.ga.gov.au/vincenty-direct
        lat_start = math.radians(53.320556)  # 53°19′14″N
        lon_start = math.radians(-1.729722)  # 001°43′47″W
        lat_end = math.radians(53.188432)  # 53°11′18″N
        lon_end = math.radians(0.133333)  # 000°08'00"E
        self.start_point = PointT([lon_start, lat_start], timestamp=pandas.Timestamp(1))
        self.end_point = PointT([lon_end, lat_end], timestamp=pandas.Timestamp(2))
        self.angle = math.radians(96.021667)  # 096°01′18″
        self.distance = 124_801  # meters

    def test_constructor(self):
        for illegal_argument in [
            0,
            0.,
            "0",
            datetime.datetime.now()
        ]:
            self.assertRaises(TypeError, PointT, [0, 0], timestamp=illegal_argument)

        try:
            PointT([0, 0], timestamp=pandas.Timestamp(0))
        except Exception:
            self.fail("Unexpected exception when invoking __init_().")

    def test_get_interpolated_point(self):
        ratio = 0.5
        # test interpolation on Earth
        interpolated_point = get_interpolated_point(self.start_point, self.end_point, ratio)
        self.assertEqual(self.start_point.timestamp, interpolated_point.timestamp)

    def test_point_copy(self):
        point = PointT([0, 0], timestamp=pandas.Timestamp(0), geo_reference_system='cartesian',
                       coordinates_unit='degrees')
        point_list = [point]
        point_copy = point.deep_copy()
        point_copy.timestamp = pandas.Timestamp(1)
        point_copy.to_latlon_()
        # changing the copy does not change the original point
        self.assertEqual(pandas.Timestamp(0), point.timestamp)
        self.assertEqual('cartesian', point.get_geo_reference_system())
        self.assertEqual('degrees', point.get_coordinates_unit())
        # the original point object is not changed
        self.assertEqual(point, point_list[0])

        # the copy has the same parameters as the original
        point_copy = point.deep_copy()
        self.assertEqual(pandas.Timestamp(0), point_copy.timestamp)
        self.assertEqual('cartesian', point_copy.get_geo_reference_system())
        self.assertEqual('degrees', point_copy.get_coordinates_unit())
