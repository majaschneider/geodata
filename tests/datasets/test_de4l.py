import datetime
import unittest

import dateutil.parser
import torch
import pandas as pd
from datasets.de4l import De4lSensorDataset
from geodata.route import Route


class TestDe4lSensorDataset(unittest.TestCase):

    def setup_dataloader(self, file_path, route_len):
        # read all lines from json (one json line corresponds to one data point)
        self.data_frame = pd.read_json(file_path, lines=True)
        # create a dataset with a certain route length, which creates routes from the data points
        self.dataset = De4lSensorDataset(self.data_frame, route_len)
        # create a dataloader that loads a single batch containing all routes
        self.dataloader = torch.utils.data.DataLoader(self.dataset, shuffle=False)

    def test__init__(self):
        route_len = 60
        self.setup_dataloader(file_path="tests/resources/test-sensor-dataset.json", route_len=route_len)
        for batch in self.dataloader:
            for route_tensor in batch["route_scaled_padded"]:
                # routes have same length
                self.assertEqual(len(route_tensor), route_len)
                # routes are scaled to [0, 1]
                route = Route.from_torch_tensor(route_tensor.detach())
                for point in route:
                    self.assertGreaterEqual(point.x_lon, 0)
                    self.assertGreaterEqual(1, point.x_lon)
                    self.assertGreaterEqual(point.y_lat, 0)
                    self.assertGreaterEqual(1, point.y_lat)

    def test_parse_date(self):
        date_string = '2021-02-16T09:45:02.000Z'
        self.assertTrue(isinstance(De4lSensorDataset.parse_date(date_string), datetime.datetime))
        date_datetime = dateutil.parser.parse(date_string)
        self.assertTrue(isinstance(De4lSensorDataset.parse_date(date_datetime), datetime.datetime))
        date_int = 20210216
        self.assertRaises(TypeError, lambda: De4lSensorDataset.parse_date(date_int))

    def test_calculate_location_bounds(self):
        self.setup_dataloader(file_path="tests/resources/test-sensor-dataset-small.json", route_len=1)
        location_bounds = De4lSensorDataset.calculate_location_bounds(self.data_frame)
        self.assertEqual((11.61, 11.62, 50.77, 50.87), location_bounds)


if __name__ == "__main__":
    unittest.main()
