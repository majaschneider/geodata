import unittest

import pandas as pd
from geodata.geodata.point_t import PointT
from geodata.geodatasets.geolife import GeoLifeDataset


class TestGeoLifeDataset(unittest.TestCase):

    def test_create_from_txt(self):
        paths = ['tests/resources/test_geolife.plt']
        dataset = GeoLifeDataset.create_from_txt(paths)

        self.assertIsInstance(dataset, GeoLifeDataset)
        self.assertEqual(
            PointT([116.318417, 39.984702], pd.Timestamp('2008-10-23 02:53:04'), coordinates_unit='degrees'),
            dataset.__getitem__(0).to_degrees()[0])
        self.assertEqual(dataset.__len__(), 1)
