"""
Provides a dataset class to import and preprocess Taxi Service Trajectory data as well as methods to import,
process and work with this dataset.
Source: https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii/data
"""
import datetime
import warnings

import torch
import numpy as np
import pandas as pd
from torch.nn.functional import one_hot
from torch.nn import ZeroPad2d
from torch.utils.data import Dataset
from de4l_geodata.geodata.route import Route, degrees_to_radians


class TaxiServiceTrajectoryDataset(Dataset):
    """
    Parses Taxi Service Trajectory dataset.
    """

    def __init__(self, data_frame, scale=False, location_bounds=None):
        """
        Initializes the dataset based on a pandas.DataFrame. Time between successive points in a route is assumed to be
        fifteen seconds. The data is sorted in ascending order by the route's start timestamp.

        Parameters
        ----------
        data_frame : pandas.DataFrame, containing the following columns:
            TRIP_ID: (String) It contains an unique identifier for each trip;
            CALL_TYPE: (char) It identifies the way used to demand this service. It may contain one of three
                possible values:
                ‘A’ if this trip was dispatched from the central;
                ‘B’ if this trip was demanded directly to a taxi driver on a specific stand;
                ‘C’ otherwise (i.e. a trip demanded on a random street).
            ORIGIN_CALL: (integer) It contains an unique identifier for each phone number which was used to demand, at
                least, one service. It identifies the trip’s customer if CALL_TYPE=’A’. Otherwise, it assumes a NULL
                value;
            ORIGIN_STAND: (integer): It contains an unique identifier for the taxi stand. It identifies the starting
                point of the trip if CALL_TYPE=’B’. Otherwise, it assumes a NULL value;
            TAXI_ID: (integer): It contains an unique identifier for the taxi driver that performed each trip;
            TIMESTAMP: (integer) Unix Timestamp (in seconds). It identifies the trip’s start;
            DAYTYPE: (char) It identifies the daytype of the trip’s start. It assumes one of three possible values:
                ‘B’ if this trip started on a holiday or any other special day (i.e. extending holidays, floating
                    holidays, etc.);
                ‘C’ if the trip started on a day before a type-B day;
                ‘A’ otherwise (i.e. a normal day, workday or weekend).
            MISSING_DATA: (Boolean) It is FALSE when the GPS data stream is complete and TRUE whenever one (or more)
                locations are missing
            POLYLINE: (String): It contains a list of GPS coordinates (i.e. WGS84 format) mapped as a string. The
                beginning and the end of the string are identified with brackets (i.e. [ and ], respectively). Each
                pair of coordinates is also identified by the same brackets as [LONGITUDE, LATITUDE]. This list contains
                one pair of coordinates for each 15 seconds of trip. The last list item corresponds to the trip’s
                destination while the first one represents its start;
        scale : bool
            If True, route points will be scaled by location_bounds.
        location_bounds : tuple
            Outer bounds of the location data's coordinates in format (longitude minimum, longitude maximum, latitude
            minimum, latitude maximum). If None, values will be calculated from data_frame.
        """
        self.scale = scale
        self.time_between_route_points = pd.Timedelta(seconds=15)

        data_frame["route"] = data_frame["POLYLINE"].copy().apply(self.route_str_to_list)
        # taxi data is provided in format lonlat and degrees, geodata.route.Route requires lonlat and radians
        data_frame["route"] = data_frame["route"].copy().apply(degrees_to_radians)

        # remove all rows that have caused polyline parsing issues
        data_frame.drop(data_frame.loc[data_frame["POLYLINE"] == "[]"].index, inplace=True)

        # add timestamp information for route and start
        data_frame["trip_time_start_utc"] = data_frame["TIMESTAMP"].copy().apply(
            lambda x: datetime.datetime.utcfromtimestamp(int(x))
        )
        data_frame = data_frame.sort_values(by=['trip_time_start_utc'])
        data_frame["timestamps"] = data_frame.copy().apply(
            lambda row: self.get_timestamps(row, self.time_between_route_points), axis=1
        )

        self.data_frame = data_frame
        self.max_route_len = self.__max_route_len__()

        if location_bounds is None:
            self.location_bounds = self.calculate_location_bounds(data_frame)
        else:
            self.location_bounds = location_bounds

    def __len__(self):
        return len(self.data_frame)

    def __max_route_len__(self):
        return self.data_frame["route"].transform(len).max()

    def __getitem__(self, idx):
        """
        Provides a data sample containing one hot encoded time features and scaled and padded route points.

        Parameters
        ----------
        idx : int
            The index of a route.

        Returns
        -------
        sample : dict
            A data sample containing 'day_of_week', 'quarter_hour_of_day', 'month', 'route', 'route_with_timestamps' and
            'route_scaled_padded'.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        timestamp_utc = self.data_frame.trip_time_start_utc.iloc[idx]
        timestamps = self.data_frame.timestamps.iloc[idx]
        route = Route(self.data_frame.route.iloc[idx])
        route_tensor_raw = torch.tensor(route, dtype=torch.float64, requires_grad=True)
        route_len = len(route)

        # initialize one hot representation
        day_of_week_one_hot = torch.zeros([route_len, 7], dtype=torch.float64)
        quarter_hour_of_day_one_hot = torch.zeros([route_len, 96], dtype=torch.float64)
        month_one_hot = torch.zeros([route_len, 12], dtype=torch.float64)

        # calculate features for each route point
        for i in range(route_len):
            day_of_week = timestamp_utc.isoweekday() - 1  # 0 - 6 = Mo - Su
            quarter_of_hour = int(np.floor(timestamp_utc.minute / 15))  # 0 - 3
            hour_of_day = timestamp_utc.hour  # 0 - 23
            quarter_hour_of_day = hour_of_day * 4 + quarter_of_hour  # 0 - 95
            month = timestamp_utc.month - 1  # 0 - 11
            # update one hot representation
            day_of_week_one_hot[i] = one_hot(torch.tensor(day_of_week), num_classes=7)
            quarter_hour_of_day_one_hot[i] = one_hot(torch.tensor(quarter_hour_of_day), num_classes=96)
            month_one_hot[i] = one_hot(torch.tensor(month), num_classes=12)
            # advance timestamp
            timestamp_utc += self.time_between_route_points

        # scale route points by location_bounds
        if self.scale:
            route.scale(self.location_bounds)

        # pad features to max_route_len
        pad_len = self.max_route_len - route_len
        pad = ZeroPad2d((0, 0, 0, pad_len))
        day_of_week_one_hot = pad(day_of_week_one_hot)
        quarter_hour_of_day_one_hot = pad(quarter_hour_of_day_one_hot)
        month_one_hot = pad(month_one_hot)
        route_tensor_raw_padded = pad(route_tensor_raw)
        route.pad(self.max_route_len)
        route_tensor_scaled_padded = torch.tensor(route, dtype=torch.float64, requires_grad=True)
        for _ in range(pad_len):
            timestamps.append(timestamps[-1])
        route_with_timestamps = Route(route, timestamps)

        sample = {
            "day_of_week": day_of_week_one_hot,
            "quarter_hour_of_day": quarter_hour_of_day_one_hot,
            "month": month_one_hot,
            "route_with_timestamps": route_with_timestamps,
            "route": route_tensor_raw_padded,
            "route_scaled_padded": route_tensor_scaled_padded,
        }

        return sample

    @classmethod
    def get_timestamps(cls, row, time_between_route_points):
        """
        Calculates the timestamp of each route point.

        Parameters
        ----------
        row : pd.core.series.Series
            The data series containing 'route' and 'trip_time_start_utc'.
        time_between_route_points : pd.Timedelta
            The time difference between consecutive stops.

        Returns
        -------
        List
            A list of timestamps corresponding to each route point.
        """
        route_length = len(row["route"])
        start_timestamp = row["trip_time_start_utc"]
        return [start_timestamp + i * time_between_route_points for i in range(route_length)]

    @classmethod
    def route_str_to_list(cls, route):
        """
        Converts route from string to list format.

        Parameters
        ----------
        route : str
            A route in string format. Example: [[-8.58, 41.14], [-8.5, 41.1]]

        Returns
        -------
        transformed : list
            A route in list format.

        """
        transformed = []
        try:
            route = route.replace(" ", "").replace("],[", "];[").replace("[[", "[").replace("]]", "]").split(";")
            for point in route:
                point = point.replace("[", "").replace("]", "").split(",")
                transformed.append([float(point[0]), float(point[1])])
        except Exception:
            warnings.warn("An error occurred during parsing of route from string to list. An empty list will be "
                          "returned.")
        return transformed

    @classmethod
    def calculate_location_bounds(cls, data_frame):
        """
        Determines the minimum and maximum values of the location coordinates from all points in all routes.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            A data frame containing at least the column 'route'. Every entry should be of type geodata.route.Route.

        Returns
        -------
        longitude_min, longitude_max, latitude_min, latitude_max : float
            The minimum and maximum location coordinates of all route points.

        """
        longitude_min = min(data_frame["route"].apply(lambda x: min([p.x_lon for p in Route(x)])))
        longitude_max = max(data_frame["route"].apply(lambda x: max([p.x_lon for p in Route(x)])))
        latitude_min = min(data_frame["route"].apply(lambda x: min([p.y_lat for p in Route(x)])))
        latitude_max = max(data_frame["route"].apply(lambda x: max([p.y_lat for p in Route(x)])))
        return longitude_min, longitude_max, latitude_min, latitude_max

    @classmethod
    def create_from_csv(cls, path, limit=None):
        """
        Initializes a TaxiServiceTrajectoryDataset by reading from a csv. If size is given, only the first lines are
        read. The csv file should at least have the columns mentioned in TaxiServiceTrajectoryDataset.__init__():
            TRIP_ID: (String)
            CALL_TYPE: (char)
            ORIGIN_CALL: (integer)
            ORIGIN_STAND: (integer)
            TAXI_ID: (integer)
            TIMESTAMP: (integer)
            DAYTYPE: (char)
            MISSING_DATA: (Boolean)
            POLYLINE: (String)

        Parameters
        ----------
        path : str
            Relative path to a csv file containing data required for a TaxiServiceTrajectoryDataset.
        limit : int
            Number of lines from the file that should be read. If None, the whole file while be read.

        Returns
        -------
        TaxiServiceTrajectoryDataset
            A TaxiServiceTrajectoryDataset created from the given csv.
        """
        assert isinstance(path, str)
        assert path[-4:] == '.csv'

        if isinstance(limit, int):
            dataloader = pd.read_csv(path, sep=',', encoding='latin1', chunksize=limit)
            data_frame = next(dataloader)
            dataloader.close()
        else:
            data_frame = pd.read_csv(path, sep=',', encoding='latin1')

        return TaxiServiceTrajectoryDataset(data_frame)
