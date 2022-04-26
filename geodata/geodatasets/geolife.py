"""
Provides a dataset class to import 'GeoLife' GPS trajectory dataset. It was collected in (Microsoft Research Asia)
Geolife project by 182 users in a period of over five years (from April 2007 to August 2012). For some users
Transportation mode labels are available.

References:
Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, Wei-Ying Ma. Understanding Mobility Based on GPS Data. In Proceedings of
ACM conference on Ubiquitous Computing (UbiComp 2008), Seoul, Korea. ACM Press: 312-321.

Yu Zheng, Xing Xie, Wei-Ying Ma, GeoLife: A Collaborative Social Networking Service among User, location and trajectory.
Invited paper, in IEEE Data Engineering Bulletin. 33, 2, 2010, pp. 32-40.
"""
import pandas as pd
from torch.utils.data import Dataset

from geodata.geodata.point_t import PointT
from geodata.geodata.route import Route


class GeoLifeDataset(Dataset):
    """Parses a GeoLife trajectory dataset."""

    def __init__(self, data_frame):
        """
        Initializes the dataset based on a pd.DataFrame. The data is sorted in ascending order by the route's start
        timestamp. The dataset might contain data collected by multiple users, covering multiple days. Each route in the
        dataset contains all points one user has collected during one day. The order of routes and days is determined by
        the order of the input data.

        Parameters
        ----------
        data_frame : pd.DataFrame, containing the following columns:
            USER_ID: (numpy.int64) A unique identifier for each user
            LAT: (float) Latitude coordinate values in degrees format
            LON: (float) Longitude coordinate values in degrees format
            DATE_TIME: (String) UTC ISO 8601 timestamps in format YYYY-MM-DD hh:mm:ss
        """
        self.data_frame_per_point = data_frame
        self.data_frame_per_route = self.create_route_based_data_frame()
        # create a Route list from 'LAT' and 'LON' and 'DATE_TIME'. Distinguish by 'USER_ID' and date of 'DATE_TIME'.
        current_user_id = int(self.data_frame_per_point['USER_ID'].iloc[0])
        current_timestamp = pd.Timestamp(self.data_frame_per_point['DATE_TIME'].iloc[0])
        route = Route()
        for _, row in self.data_frame_per_point.iterrows():
            next_user_id = int(row['USER_ID'])
            next_timestamp = pd.Timestamp(row['DATE_TIME'])
            if not (next_user_id == current_user_id and next_timestamp.date() == current_timestamp.date()):
                self.add_route(current_user_id, current_timestamp.date(), route.to_radians(), route.get_timestamps())
                current_user_id = next_user_id
                current_timestamp = next_timestamp
                route = Route()
            try:
                point_t = PointT([row['LON'], row['LAT']], timestamp=next_timestamp, coordinates_unit='degrees')
                route.append(point_t)
            except:
                print(f"A point could not be created ([{row['LON']}, {row['LAT']}] in degrees) and is ignored.")
        self.add_route(current_user_id, current_timestamp.date(), route.to_radians(), route.get_timestamps())
        self.data_frame_per_route = self.data_frame_per_route.reset_index()

    def add_route(self, user_id, date, route, timestamps):
        route_data = {'user_id': [user_id], 'date': [date], 'route': [route], 'timestamps': [timestamps]}
        self.data_frame_per_route = pd.concat([self.data_frame_per_route, pd.DataFrame(data=route_data)])

    @classmethod
    def create_route_based_data_frame(cls):
        """
        Create an empty dataframe for storing route-based user data with the following columns:
        user_id : str
            The id of the user.
        date : datetime64
            The start_point time of the trajectory in utc format.
        route : Route
            The trajectory of the user.

        Returns
        -------
        df : pd.Dataframe
            An empty dataframe for storing route-based user trajectory data.
        """
        columns = {'user_id': pd.Series(dtype='str'),
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
        Creates a GeoLife dataset from one or multiple txt file data sources. Uses pandas to read and process the files.

        Parameters
        ----------
        list_of_paths_to_datasets : list
            A list of file paths to the t-drive data files.
        limit : int
            An optional limitation in length for the resulting dataset. Only the first 'limit' entries of the provided
            files will be processed.

        Returns
        -------
        dataset : GeoLifeDataset
            A GeoLifeDataset created from the provided data files.
        """
        column_names = ['LAT', 'LON', 'ignore1', 'ignore2', 'ignore3', 'DATE', 'TIME']
        data_frames = []
        user_id = 0
        for path in list_of_paths_to_datasets:
            df = pd.read_csv(path, sep=',', skiprows=6, nrows=limit, header=None, names=column_names,
                             encoding='utf-8-sig')
            df['USER_ID'] = user_id
            df = df.drop(['ignore1'], axis=1)
            df = df.drop(['ignore2'], axis=1)
            df = df.drop(['ignore3'], axis=1)
            df['DATE_TIME'] = df.apply(lambda row: pd.Timestamp(row['DATE'] + ' ' + row['TIME']), axis=1)
            user_id += 1
            data_frames.append(df)

        data_frame = pd.concat(data_frames)

        return GeoLifeDataset(data_frame)
