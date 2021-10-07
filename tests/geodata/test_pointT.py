import unittest
import math

from geodata.point_t import PointT, get_interpolated_point


class TestPointMethods(unittest.TestCase):
    def setUp(self):
        # See example here: http://www.movable-type.co.uk/scripts/latlong.html
        # Values vary slightly due to the use of different earth_radius values
        # https://geodesyapps.ga.gov.au/vincenty-direct
        lat_start = math.radians(53.320556)  # 53°19′14″N
        lon_start = math.radians(-1.729722)  # 001°43′47″W
        lat_end = math.radians(53.188432)  # 53°11′18″N
        lon_end = math.radians(0.133333)  # 000°08'00"E
        self.start_point = PointT([lon_start, lat_start], timestamp=1)
        self.end_point = PointT([lon_end, lat_end], timestamp=2)
        self.angle = math.radians(96.021667)  # 096°01′18″
        self.distance = 124_801  # meters

    def test_get_interpolated_point(self):
        ratio = 0.5
        # test interpolation on Earth
        interpolated_point = get_interpolated_point(self.start_point, self.end_point, ratio)
        self.assertEqual(self.start_point.timestamp, interpolated_point.timestamp)
