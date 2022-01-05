import math
import unittest
import torch
import pandas as pd
from de4l_geodata.geodatasets.taxi import TaxiServiceTrajectoryDataset
from de4l_geodata.geodata.route import Route


class TestTaxiServiceTrajectoryDataset(unittest.TestCase):
    def setUp(self):
        file_path = "tests/resources/test-taxi-dataset.csv"
        # read the full dataset
        data_frame = pd.read_csv(file_path, sep=",", encoding="latin1")
        self.dataset = TaxiServiceTrajectoryDataset(data_frame, scale=True)
        nr_routes = len(self.dataset)
        # batch the full dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=nr_routes)

    def test__max_route_len__(self):
        self.assertEqual(612, self.dataset.__max_route_len__())

    def test__init__(self):
        for batch in self.dataloader:
            for route_tensor in batch["route_scaled_padded"]:
                route = Route.from_torch_tensor(route_tensor.detach())
                # routes are of max route length
                self.assertEqual(len(route), self.dataset.max_route_len)
                # routes are scaled to [0, 1]
                for point in route:
                    self.assertGreaterEqual(point.x_lon, 0)
                    self.assertGreaterEqual(1, point.x_lon)
                    self.assertGreaterEqual(point.y_lat, 0)
                    self.assertGreaterEqual(1, point.y_lat)

    def test_route_str_to_list(self):
        route_list = [[-8.5, 41.14], [-8.58, 41.14]]
        route_str = str(route_list)
        self.assertEqual(route_list, self.dataset.route_str_to_list(route_str))

    def test_calculate_location_bounds(self):
        file_path = "tests/resources/test-taxi-dataset-small.csv"
        data_frame = pd.read_csv(file_path, sep=",", encoding="latin1")
        dataset = TaxiServiceTrajectoryDataset(data_frame)
        location_bounds = (math.radians(-8.61), math.radians(-8.5), math.radians(41.1), math.radians(41.7))
        self.assertEqual(location_bounds, dataset.location_bounds)

    def test_create_from_csv(self):
        path = "tests/resources/test-taxi-dataset.csv"

        correct_path_dataset = TaxiServiceTrajectoryDataset.create_from_csv(path)
        self.assertTrue(isinstance(correct_path_dataset, TaxiServiceTrajectoryDataset))

        correct_path_dataset_small = TaxiServiceTrajectoryDataset.create_from_csv(path, limit=100)
        self.assertEqual(correct_path_dataset_small.data_frame.shape[0], 100)

        # test wrong file type by slicing the last character off
        self.assertRaises(AssertionError, lambda: TaxiServiceTrajectoryDataset.create_from_csv(path[:-1]))

        # test wrong path by slicing the first character off
        self.assertRaises(FileNotFoundError, lambda: TaxiServiceTrajectoryDataset.create_from_csv(path[1:]))
