"""
Provides a dataset class to import 'T-Drive' Taxi Trajectory data as well as methods to import,
process and work with this dataset.
The source is a dataset with taxi data collected by Microsoft in Beijing in 2008.
"""
import pandas as pd
from torch.utils.data import Dataset

from geodata.geodata.point_t import PointT
from geodata.geodata.route import Route


class TDriveDataset(Dataset):
    """Parses a T-Drive trajectory dataset and makes routes available."""

    def __init__(self, data_frame):
        """
        Initializes the dataset based on a pd.DataFrame. The data is sorted in ascending order by the route's start
        timestamp. The dataset might contain data collected by multiple taxis, covering multiple days. Each route in the
        dataset contains all points one taxi has collected during one day. The order of routes and days is determined by
        the order of the input data.

        Parameters
        ----------
        data_frame : pd.DataFrame, containing the following columns:
            TAXI_ID: (numpy.int64) A unique identifier for each taxi
            DATE_TIME: (String) UTC ISO 8601 timestamps in format YYYY-MM-DD hh:mm:ss
            LON: (float) Longitude coordinate values in degrees format
            LAT: (float) Latitude coordinate values in degrees format
        """
        self.data_frame_per_point = data_frame
        self.data_frame_per_route = self.create_route_based_data_frame()
        # create a Route list from 'LON' and 'LAT' and 'DATE_TIME'. Distinguish by 'TAXI_ID' and date of 'DATE_TIME'.
        current_taxi_id = int(self.data_frame_per_point['TAXI_ID'].iloc[0])
        current_timestamp = pd.Timestamp(self.data_frame_per_point['DATE_TIME'].iloc[0])
        route = Route()
        for _, row in self.data_frame_per_point.iterrows():
            next_taxi_id = int(row['TAXI_ID'])
            next_timestamp = pd.Timestamp(row['DATE_TIME'])
            if not (next_taxi_id == current_taxi_id and next_timestamp.date() == current_timestamp.date()):
                self.add_route(current_taxi_id, current_timestamp.date(), route.to_radians(), route.get_timestamps())
                current_taxi_id = next_taxi_id
                current_timestamp = next_timestamp
                route = Route()
            try:
                point_t = PointT([row['LON'], row['LAT']], timestamp=next_timestamp, coordinates_unit='degrees')
                route.append(point_t)
            except:
                print(f"A point could not be created ([{row['LON']}, {row['LAT']}] in degrees) and is ignored.")
        self.add_route(current_taxi_id, current_timestamp.date(), route.to_radians(), route.get_timestamps())
        self.data_frame_per_route = self.data_frame_per_route.reset_index()

    def add_route(self, taxi_id, date, route, timestamps):
        route_data = {'taxi_id': [taxi_id], 'date': [date], 'route': [route], 'timestamps': [timestamps]}
        self.data_frame_per_route = pd.concat([self.data_frame_per_route, pd.DataFrame(data=route_data)])

    @classmethod
    def create_route_based_data_frame(cls):
        """
        Create an empty dataframe for storing route-based taxi data with the following columns:
        taxi_id : str
            The id of the taxi.
        date : datetime64
            The start_point time of the trajectory in utc format.
        route : Route
            The trajectory of the taxi.

        Returns
        -------
        df : pd.Dataframe
            An empty dataframe for storing route-based taxi data.
        """
        columns = {'taxi_id': pd.Series(dtype='str'),
                   'date': pd.Series(dtype='datetime64[ns]'),
                   'route': None,
                   'timestamps': None}
        df = pd.DataFrame(columns)
        df['route'].map(Route)
        df['timestamps'].map(list)
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

    @classmethod
    def create_from_txt(cls, list_of_paths_to_datasets, limit=None):
        """
        Creates a T-Drive dataset from one or multiple txt file data sources. Uses pandas to read and process the files.

        Parameters
        ----------
        list_of_paths_to_datasets : list
            A list of file paths to the t-drive data files.
        limit : int
            An optional limitation in length for the resulting dataset. Only the first 'limit' entries of the provided
            files will be processed.

        Returns
        -------
        dataset : TDriveDataset
            A TDriveDataset created from the provided data files.
        """
        column_names = ['TAXI_ID', 'DATE_TIME', 'LON', 'LAT']
        data_frames = []
        for path in list_of_paths_to_datasets:
            if isinstance(limit, int):
                dataloader = pd.read_csv(path,
                                         sep=',',
                                         header=None,
                                         names=column_names,
                                         chunksize=limit,
                                         encoding='utf-8-sig')
                single_df = next(dataloader)
                dataloader.close()
            else:
                single_df = pd.read_csv(path,
                                        sep=',',
                                        header=None,
                                        names=column_names,
                                        encoding='utf-8-sig')
            data_frames.append(single_df)

        data_frame = pd.concat(data_frames)

        return TDriveDataset(data_frame)
