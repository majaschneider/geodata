import math
import unittest
import torch
import pandas as pd
from geodata.geodatasets.taxi import TaxiServiceTrajectoryDataset
from geodata.geodata.route import Route


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

    def test_calculate_location_bounds(self):
        file_path = "tests/resources/test-taxi-dataset-small.csv"
        data_frame = pd.read_csv(file_path, sep=",", encoding="latin1")
        dataset = TaxiServiceTrajectoryDataset(data_frame, max_allowed_speed_kmh=120)
        location_bounds = (math.radians(-8.610876000000001), math.radians(-8.585676),
                           math.radians(41.14557), math.radians(41.148638999999996))
        self.assertEqual(location_bounds, dataset.location_bounds)

    def test_create_from_csv(self):
        path = "tests/resources/test-taxi-dataset.csv"

        correct_path_dataset = TaxiServiceTrajectoryDataset.create_from_csv(path)
        self.assertTrue(isinstance(correct_path_dataset, TaxiServiceTrajectoryDataset))

        correct_path_dataset_small = TaxiServiceTrajectoryDataset.create_from_csv(path, nrows=100,
                                                                                  max_allowed_speed_kmh=999)
        self.assertEqual(100, correct_path_dataset_small.data_frame.shape[0])

        # test wrong file type by slicing the last character off
        self.assertRaises(AssertionError, lambda: TaxiServiceTrajectoryDataset.create_from_csv(path[:-1]))

        # test wrong path by slicing the first character off
        self.assertRaises(FileNotFoundError, lambda: TaxiServiceTrajectoryDataset.create_from_csv(path[1:]))

    def test_create_from_csv_within_time_range(self):
        path = "tests/resources/test-taxi-dataset-small.csv"
        start_date = '2013-07-01'
        end_date = '2013-07-01'
        dataset = TaxiServiceTrajectoryDataset.create_from_csv_within_time_range(
            path, start_date=start_date, end_date=end_date, max_allowed_speed_kmh=200
        )
        self.assertEqual(2, dataset.__len__())

    def test_max_speed(self):
        file_path = "tests/resources/test-taxi-dataset.csv"
        data_frame = pd.read_csv(file_path, sep=",", encoding="latin1")
        max_allowed_speed_kmh = 30
        dataset = TaxiServiceTrajectoryDataset(data_frame, scale=True, max_allowed_speed_kmh=max_allowed_speed_kmh)
        time_between_route_points = dataset.time_between_route_points
        for idx, row in dataset.data_frame.iterrows():
            route = Route(row['route'])
            self.assertLessEqual(route.max_speed(time_between_route_points), max_allowed_speed_kmh)

        max_allowed_speed_kmh = None
        data_frame = pd.read_csv(file_path, sep=",", encoding="latin1")
        dataset = TaxiServiceTrajectoryDataset(data_frame, scale=True, max_allowed_speed_kmh=max_allowed_speed_kmh)
        self.assertEqual(320, len(dataset))

    def test_error_cleaning(self):
        # Rows with empty 'Polyline' are dropped
        file_path = "tests/resources/test-taxi-dataset-big.csv"
        data_frame = pd.read_csv(file_path, sep=",", encoding="latin1")
        dataset = TaxiServiceTrajectoryDataset(data_frame, scale=True)
        self.assertEqual(4984, len(dataset))

        # Rows where 'Missing_data' is true are dropped
        file_path = "tests/resources/test-taxi-dataset-missing-data.csv"
        data_frame = pd.read_csv(file_path, sep=",", encoding="latin1")
        dataset = TaxiServiceTrajectoryDataset(data_frame, scale=True, max_allowed_speed_kmh=99999)
        self.assertEqual(98, len(dataset))
