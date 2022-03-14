"""
Provide a dataset class to import 'Cabspotting' Taxi Trajectory data as well as methods to import, process and work
with this dataset.
"""
import datetime as dt

import pandas as pd
from torch.utils.data import Dataset

from de4l_geodata.geodata.point_t import PointT
from de4l_geodata.geodata.route import Route


class CabspottingDataset(Dataset):
    """Parse a Cabspotting trajectory dataset and make routes available.
    """
    def __init__(self, data_frame):
        """
        Initialize the dataset based on a pd.DataFrame which should contain data of one single taxi, where each row
        corresponds to one data point. The data will be sorted in ascending order by timestamp.

        Parameters
        ----------
        data_frame : pd.DataFrame, containing the following columns:
            TAXI_ID: (numpy.int64) A unique identifier of the taxi
            LAT: (float) Latitude coordinate values in degrees format
            LON: (float) Longitude coordinate values in degrees format
            OCCUPANCY: (numpy.int64) An indicator if the taxi carried a customer
            DATE_TIME: (String) UTC ISO 8601 timestamps in format YYYY-MM-DD hh:mm:ss
        """
        self.data_frame_per_point = data_frame
        self.data_frame_per_route = self.create_stop_based_data_frame()
        current_taxi_id = int(self.data_frame_per_point['TAXI_ID'].iloc[0])
        current_timestamp = pd.Timestamp(dt.datetime.utcfromtimestamp(self.data_frame_per_point['DATE_TIME'].iloc[0]))
        current_occupancy = int(self.data_frame_per_point['OCCUPANCY'].iloc[0])
        route = Route()
        stops = []
        point_idx = 0
        for _, row in self.data_frame_per_point.iterrows():
            next_taxi_id = int(row['TAXI_ID'])
            next_timestamp = pd.Timestamp(dt.datetime.utcfromtimestamp(row['DATE_TIME']))
            next_occupancy = int(row['OCCUPANCY'])
            # If taxi id or date is different, create a new route
            if next_taxi_id != current_taxi_id or next_timestamp.date() != current_timestamp.date():
                self.add_route(current_taxi_id, current_timestamp.date(), route.to_radians(), route.get_timestamps(),
                               stops)
                route = Route()
                stops = []
                point_idx = 0
                current_taxi_id = next_taxi_id
                current_timestamp = next_timestamp
                current_occupancy = next_occupancy
            if current_occupancy != next_occupancy:
                # customer is being picked up
                if current_occupancy == 0:
                    stops.append(point_idx)
                # customer has been dropped off at the last point
                else:
                    stops.append(point_idx - 1)
                current_occupancy = next_occupancy
            point_t = PointT([row['LON'], row['LAT']], timestamp=next_timestamp, coordinates_unit='degrees')
            route.append(point_t)
            point_idx += 1
        self.add_route(current_taxi_id, current_timestamp.date(), route.to_radians(), route.get_timestamps(), stops)
        self.data_frame_per_route = self.data_frame_per_route.reset_index()

    def add_route(self, taxi_id, date, route, timestamps, stops):
        route_data = {'taxi_id': [taxi_id], 'date': [date], 'route': [route], 'timestamps': [timestamps],
                      'stops': [stops]}
        self.data_frame_per_route = pd.concat([self.data_frame_per_route, pd.DataFrame(data=route_data)])

    @classmethod
    def create_stop_based_data_frame(cls):
        """
        Create an empty dataframe for storing stop-based taxi data with the following columns:
        taxi_id : str
            The id of the taxi.
        date : datetime64
            The start_point time of the trajectory in utc format.
        route : Route
            The trajectory of the taxi.
        stops : Route
            The customer pick-up or delivery stops of each taxi, inferred from the endpoints of each trip.

        Returns
        -------
        df : pd.Dataframe
            An empty dataframe for storing stop-based taxi data.
        """
        columns = {'taxi_id': pd.Series(dtype='str'),
                   'date': pd.Series(dtype='datetime64[ns]'),
                   'route': None,
                   'timestamps': None,
                   'stops': None}
        df = pd.DataFrame(columns)
        df['route'].map(Route)
        df['timestamps'].map(list)
        df['stops'].map(list)
        return df

    def __len__(self):
        return len(self.data_frame_per_route)

    def __getitem__(self, idx):
        """
        Provides a route containing points with timestamps.

        Parameters
        ----------
        idx : int
            The index of a route in the dataset.

        Returns
        -------
        route : Route
            A route containing points with timestamps.
        """
        return Route(self.data_frame_per_route['route'].iloc[idx])

    def get_stops(self, idx):
        return self.data_frame_per_route['stops'].iloc[idx]

    @classmethod
    def create_from_txt(cls, list_of_paths_to_datasets, limit=None):
        """
        Creates a Cabspotting dataset from one or multiple txt file data sources. Uses pandas to read and process the
        files.

        Parameters
        ----------
        list_of_paths_to_datasets : list
            A list of file paths to the data files.
        limit : int
            The number of rows to read from the file. Data after limit will be skipped.

        Returns
        -------
        dataset : CabspottingDataset
            A CabspottingDataset created from the provided data files.
        """
        column_names = ['LAT', 'LON', 'OCCUPANCY', 'DATE_TIME']
        data_frames = []
        taxi_id = 0
        for path in list_of_paths_to_datasets:
            if isinstance(limit, int):
                dataloader = pd.read_csv(path,
                                         sep=' ',
                                         header=None,
                                         names=column_names,
                                         chunksize=limit,
                                         encoding='utf-8-sig')
                single_df = next(dataloader)
                dataloader.close()
            else:
                single_df = pd.read_csv(path,
                                        sep=' ',
                                        header=None,
                                        names=column_names,
                                        encoding='utf-8-sig')
            single_df['TAXI_ID'] = taxi_id
            taxi_id += 1
            data_frames.append(single_df)

        data_frame = pd.concat(data_frames)

        return CabspottingDataset(data_frame)
