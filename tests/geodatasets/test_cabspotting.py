import unittest
import datetime as dt
import os

import pandas as pd
from de4l_geodata.geodata.point_t import PointT
from de4l_geodata.geodatasets.cabspotting import CabspottingDataset


class TestCabspottingDataset(unittest.TestCase):

    def test_create_from_txt(self):
        base_path = './'
        paths = [base_path + 'tests/resources/test_cabspotting_1.txt',
                 base_path + 'tests/resources/test_cabspotting_2.txt']
        dataset = CabspottingDataset.create_from_txt(paths)

        self.assertIsInstance(dataset, CabspottingDataset)
        self.assertEqual(2, dataset.__len__())
        self.assertEqual(
            PointT([-122.1343, 37.45667], pd.Timestamp(dt.datetime(2008, 6, 10, 9, 1, 2)), coordinates_unit='degrees'),
            dataset.__getitem__(0).to_degrees()[0]
        )
        self.assertEqual([1, 2], dataset.data_frame_per_route['stops'].iloc[0])
