import unittest

import pandas as pd
from de4l_geodata.geodata.point_t import PointT
from de4l_geodata.geodatasets.t_drive import TDriveDataset


class TestTDriveDataset(unittest.TestCase):

    def test_create_from_txt(self):
        paths = ['tests/resources/test-t-drive-dataset-1.txt', 'tests/resources/test-t-drive-dataset-2.txt']
        dataset = TDriveDataset.create_from_txt(paths)

        self.assertIsInstance(dataset, TDriveDataset)
        self.assertEqual(PointT([23.53423, 34.39873], pd.Timestamp('2022-02-02 12:34:12'), coordinates_unit='degrees'),
                         dataset.__getitem__(0).to_degrees()[0])
        self.assertEqual(dataset.__len__(), 5)

        # missing taxi id
        paths = ['tests/resources/test-t-drive-dataset-2.txt', 'tests/resources/test-t-drive-dataset-3.txt']
        self.assertRaises(ValueError,
                          TDriveDataset.create_from_txt,
                          paths)

        # empty file
        path = ['tests/resources/test-t-drive-dataset-4.txt']
        self.assertRaises(IndexError,
                          TDriveDataset.create_from_txt,
                          path)

        # file not existing
        path = ['tests/resources/test-t-drive-dataset-5.txt']
        self.assertRaises(FileNotFoundError,
                          TDriveDataset.create_from_txt,
                          path)
