"""Provides a class for reading and preprocessing data from project DE4L.
"""

import datetime
from math import radians

import dateutil.parser
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

from geodata.point_t import PointT
from geodata.point import Point
from geodata.route import Route


class De4lSensorDataset(Dataset):
    """
    Parses points and timestamps from air beam sensor data of project DE4L and runs the following preprocessing routines
    to prepare the data as input for machine learning systems:
        - scales coordinates to [0,1]
        - pads routes with zero values to be of same length
        - applies one hot encoding for columns containing temporal information
    """

    # todo: use route id as identifier for a route when sampling instead of assuming a fixed sequence length
    def __init__(self, data_frame, route_len, location_bounds=None):
        """
        De4lSensorDataset contains single points (route information has to be deduced from timestamp and driver id).
        Routes will be created by sampling successive points up to route_len.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            A data frame containing the sensor data and specifying at least the columns
                'timestamp' UTC in format ISO 8601
                'location' (dict('lon', 'lat'))
        route_len : int
            A fixed length per route.
        location_bounds : tuple
            Outer bounds of the location data's coordinates in format (longitude minimum, longitude maximum, latitude
            minimum, latitude maximum). If None, values will be calculated from data_frame.
        """
        self.route_len = route_len

        if location_bounds is None:
            self.location_bounds = self.calculate_location_bounds(data_frame)
        else:
            self.location_bounds = location_bounds

        data_frame["timestamp"] = data_frame["timestamp"].apply(self.parse_date)

        # day of the week from 0 to 6
        data_frame["day_of_week"] = data_frame["timestamp"].apply(lambda x: x.isoweekday() - 1)
        # quarter of an hour from 0 to 95 (4 quarters per hour times 24 hours per day)
        data_frame["quarter_hour_of_day"] = data_frame["timestamp"].apply(
            lambda x: x.hour * 4 + int(np.floor(x.minute / 15))
        )
        # month of the year from 0 to 11
        data_frame["month"] = data_frame["timestamp"].apply(lambda x: x.month - 1)
        self.data_frame = data_frame

    def __len__(self):
        return int(np.ceil(len(self.data_frame) / self.route_len))

    def __getitem__(self, idx):
        """
        Provides a data sample containing one hot encoded time features and routes with timestamps as well as scaled and
        padded routes.

        Parameters
        ----------
        idx : int
            The index of a route.

        Returns
        -------
        sample : dict
            A data sample containing 'day_of_week', 'quarter_hour_of_day', 'month', 'route_with_timestamps',
            'route_tensor_raw_padded' and 'route_tensor_scaled_padded'.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # indexing like this only works because all routes have the same fixed length
        if idx > 0:
            start_idx = idx * self.route_len
        else:
            start_idx = idx

        day_of_week_one_hot = torch.zeros([self.route_len, 7], dtype=torch.float64)
        quarter_hour_of_day_one_hot = torch.zeros([self.route_len, 96], dtype=torch.float64)
        month_one_hot = torch.zeros([self.route_len, 12], dtype=torch.float64)

        route_idx = 0
        route = Route()
        route_with_timestamps = Route()
        for i in range(start_idx, min(start_idx + self.route_len, len(self.data_frame))):
            location = self.data_frame.loc[i, "location"]
            timestamp = self.data_frame.loc[i, "timestamp"]
            point = Point([radians(location["lon"]), radians(location["lat"])])
            route.append(point)
            point_with_timestamp = PointT(point, timestamp=timestamp)
            route_with_timestamps.append(point_with_timestamp)

            # one hot encoding
            day_of_week_one_hot[route_idx] = one_hot(torch.tensor(self.data_frame.loc[i, "day_of_week"]), num_classes=7)
            quarter_hour_of_day_one_hot[route_idx] = one_hot(
                torch.tensor(self.data_frame.loc[i, "quarter_hour_of_day"]),
                num_classes=96
            )
            month_one_hot[route_idx] = one_hot(torch.tensor(self.data_frame.loc[i, "month"]), num_classes=12)
            route_idx += 1

        route_raw_padded = Route(route)
        route_raw_padded.pad(self.route_len)
        route_tensor_raw_padded = torch.tensor(route_raw_padded, dtype=torch.float64, requires_grad=True)

        route_scaled_padded = Route(route)
        route_scaled_padded.scale(self.location_bounds)
        route_scaled_padded.pad(self.route_len)
        route_tensor_scaled_padded = torch.tensor(route_scaled_padded, dtype=torch.float64, requires_grad=True)

        sample = {
            "day_of_week": day_of_week_one_hot,
            "quarter_hour_of_day": quarter_hour_of_day_one_hot,
            "month": month_one_hot,
            # if any other route format is required, it can be added here
            "route_with_timestamps": route_with_timestamps,
            "route_tensor_raw_padded": route_tensor_raw_padded,
            "route_tensor_scaled_padded": route_tensor_scaled_padded,
        }

        return sample

    @classmethod
    def parse_date(cls, date):
        """
        Convert a string to a date object. Already parsed objects are ignored.

        Parameters
        ----------
        date : str or datetime.datetime
            A date that is either represented as a string or already parsed as datetime object.

        Returns
        -------
        date : datetime.datetime
            A datetime object.
        """
        try:
            parsed_date = dateutil.parser.parse(date)
            return parsed_date
        except TypeError as error:
            if isinstance(date, datetime.datetime):
                return date
            else:
                raise TypeError("A timestamp could not be parsed. Please check for correct format.") from error

    @classmethod
    def calculate_location_bounds(cls, data_frame):
        """
        Determine the minimum and maximum values of the location coordinates from all points in all routes.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            A data frame containing at least the column 'location' which must consist of dicts with 'lon' and 'lat' as
            keys and numerical values.

        Returns
        -------
        longitude_min, longitude_max, latitude_min, latitude_max : float
            The minimum and maximum location coordinates of all route points.
        """
        longitude_min = min(data_frame["location"].apply(lambda x: radians(x["lon"])))
        longitude_max = max(data_frame["location"].apply(lambda x: radians(x["lon"])))
        latitude_min = min(data_frame["location"].apply(lambda x: radians(x["lat"])))
        latitude_max = max(data_frame["location"].apply(lambda x: radians(x["lat"])))
        return longitude_min, longitude_max, latitude_min, latitude_max

    @classmethod
    def create_from_json(cls, path, route_len, limit=None):
        """
        Initializes a dataset by reading from a json. If limit is given, only the first lines are read. The json file
        should contain a list of entries so that it is possible to use 'lines' as parameter for read_json().
        Each entry in the json file should at least have the following key-value-pairs:
            'timestamp' UTC in format ISO 8601
            'location' (dict('lon', 'lat'))

        Parameters
        ----------
        path : str
            Relative path to a json file containing data entries required for a De4lSensorDataset.
        route_len : int
            Fixed length per route.
        limit : int
            Maximum number of lines from the file that should be read. If None, the whole file is read.

        Returns
        -------
        De4lSensorDataset
            A De4lSensorDataset created from the given json.
        """
        assert isinstance(path, str)
        assert path[-5:] == '.json'

        if isinstance(limit, int):
            dataloader = pd.read_json(path, lines=True, chunksize=limit)
            data_frame = next(dataloader)
            dataloader.close()
        else:
            data_frame = pd.read_json(path, lines=True)

        return De4lSensorDataset(data_frame, route_len)
