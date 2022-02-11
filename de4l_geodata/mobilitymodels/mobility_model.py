import warnings

import openrouteservice
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np

from de4l_geodata.geodata.route import Route
from de4l_geodata.geodata.point import Point, get_distance
from de4l_geodata.helper import parser


def calculate_mobility_model(df, nominatim, transition_df=None, location_df=None):
    """
    Calculates transition probabilities between street segments per weekday and an assignment of locations
    to openstreetmap segment ids.

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        A dataframe holding mobility data, which contains the following columns:
        trip_time_start_utc : pd.Timestamp
            The start timestamp of a mobility trace.
        route : Route
            The mobility trace in radians format.
    nominatim : Nominatim
        A running instance of Nominatim.
    transition_df : pd.core.frame.DataFrame
        The data frame holding existing transition probabilities.
    location_df : pd.core.frame.DataFrame
        The data frame holding existing location information for osm_ids.

    Returns
    -------
    transition_df : pd.core.frame.DataFrame
        The dataframe containing the transition probabilities between the street segment ids, derived from df.
    location_df : pd.core.frame.DataFrame
        The dataframe containing the transition probabilities between the street segment ids, derived from df.
    """

    df = df.copy()
    df['weekday'] = df['trip_time_start_utc'].dt.weekday

    if transition_df is None:
        transition_df = create_transition_df()
    if location_df is None:
        location_df = create_location_df()

    for _, row in df.iterrows():
        route = Route(row['route'])
        weekday = row['weekday']
        route_len = len(route)

        for i in range(route_len - 1):
            current_point = route[i].to_degrees(ignore_warning=True)
            next_point = route[i + 1].to_degrees(ignore_warning=True)
            try:
                # use default zoom level of 18 to get 'way' level = street segment
                current_osm_id = nominatim.reverse([current_point.y_lat, current_point.x_lon]).raw['osm_id']
                next_osm_id = nominatim.reverse([next_point.y_lat, next_point.x_lon]).raw['osm_id']
                # todo: replace
                distance, duration, _ = get_shortest_route_details(Route([current_point, next_point]))
                existing_transition = find_transition(current_osm_id, next_osm_id, weekday, transition_df)

                if existing_transition.empty:
                    current_transition = \
                        pd.DataFrame({'osm_id_1': [current_osm_id], 'osm_id_2': [next_osm_id], 'weekday': [weekday],
                                      'count': [1], 'distance_sum': distance, 'duration_sum': duration})
                    transition_df = pd.concat([transition_df, current_transition], ignore_index=True)
                else:
                    transition_df.loc[existing_transition.index, 'count'] = existing_transition['count'] + 1
                    transition_df.loc[existing_transition.index, 'distance_sum'] = \
                        existing_transition['distance_sum'] + distance
                    transition_df.loc[existing_transition.index, 'duration_sum'] = \
                        existing_transition['duration_sum'] + duration

                location_df = update_osm_locations(current_osm_id, location_df, current_point)
                if i == route_len - 2:
                    location_df = update_osm_locations(next_osm_id, location_df, next_point)
            except Exception:
                print(f"osm id could not be retrieved for points {current_point} and {next_point}")
        transition_df['distance_avg'] = transition_df['distance_sum'] / transition_df['count']
        transition_df['duration_avg'] = transition_df['duration_sum'] / transition_df['count']
        transition_df['sum_of_count_osm_id_1'] = \
            transition_df.groupby(['osm_id_1', 'weekday'])['count'].transform(np.sum)
        transition_df['transition_probability'] = transition_df['count'] / transition_df['sum_of_count_osm_id_1']
        location_df['center'] = location_df['locations'].apply(lambda x: Route(x).get_average_point())
    return transition_df, location_df


def find_transition(osm_id_1, osm_id_2, weekday, df):
    """
    Finds the row in a data frame describing the transition from osm_id_1 to osm_id_2 on a certain weekday.

    Parameters
    ----------
    osm_id_1 : int
        The id of the openstreetmap road segment of the transition start.
    osm_id_2 : int
        The id of the openstreetmap road segment of the transition end.
    weekday : int
        The weekday of the transition.
    df : pd.core.frame.DataFrame
        The data frame in which to search for a transition.

    Returns
    -------
    The row from df of the transition described by the given parameters. Will be empty, if no transition found.
    """
    return df[(df['osm_id_1'] == osm_id_1) & (df['osm_id_2'] == osm_id_2) & (df['weekday'] == weekday)]


def find_location(osm_id, df):
    """
    Finds the row in a data frame describing the location assigned to osm_id.

    Parameters
    ----------
    osm_id : int
        The id of the searched openstreetmap road segment.
    df : pd.core.frame.DataFrame
        The data frame in which to search for a location.

    Returns
    -------
    The row from df of the location described by the given parameters. Will be empty, if no location found.
    """
    return df[df['osm_id'] == osm_id]


def update_osm_locations(osm_id, df, point):
    """
    Updates the data frame in which assignments between locations and osm_id are stored.

    Parameters
    ----------
    osm_id : int
        The id of the searched openstreetmap road segment.
    df : pd.core.frame.DataFrame
        The data frame in which to search for the osm_id.
    point : Point
        A point that belongs to the given osm_id and which will be used for updating the data frame.

    Returns
    -------
    df : pd.core.frame.DataFrame
        The updated data frame.
    """
    existing_osm_id = find_location(osm_id, df)
    if existing_osm_id.empty:
        new_osm_location = pd.DataFrame([{'osm_id': osm_id, 'locations': [point]}])
        df = pd.concat([df, new_osm_location], ignore_index=True)
    else:
        df.loc[existing_osm_id.index.values[0], 'locations'].append(point)
    return df


def create_transition_df():
    """
    Creates an empty data frame to store transitions.

    Returns
    -------
    pd.core.frame.DataFrame
        A data frame to store transitions.
    """
    return pd.DataFrame(columns={'osm_id_1': int, 'osm_id_2': int, 'weekday': int, 'count': int, 'distance_sum': float,
                                 'duration_sum': float})


def create_location_df():
    """
    Creates an empty data frame to store locations.

    Returns
    -------
    pd.core.frame.DataFrame
        A data frame to store locations.
    """
    return pd.DataFrame(columns={'osm_id': int, 'locations': None})


def location_df_from_csv(file_path):
    """
    Reads a data frame holding location-osm_id assignments from a file in csv format.

    Parameters
    ----------
    file_path : str
        The path including the file name and its file ending to the csv-file holding location and osm_id assignments.

    Returns
    -------
    pd.core.frame.DataFrame
        A data frame with locations and osm_ids loaded from the csv.
    """
    location_df = pd.read_csv(file_path, sep=',', encoding='latin1', index_col=0)
    location_df['center'] = location_df['center'].apply(
        lambda x: Point(parser.route_str_to_list(x)[0], coordinates_unit='degrees')
    )
    location_df['locations'] = location_df['locations'].apply(
        lambda x: Route(parser.route_str_to_list(x), coordinates_unit='degrees')
    )
    return location_df


def load_mobility_data_frames(path_to_transition_file, path_to_location_file):
    """
    Loads the transition and location-osm_id assignment data frames.

    Parameters
    ----------
    path_to_transition_file : str
        Path to transition file including file name and file ending.
    path_to_location_file : str
        Path to location file including file name and file ending.

    Returns
    -------
    transition_df : pd.core.frame.DataFrame
        A data frame holding transition information.
    location_df : pd.core.frame.DataFrame
        A data frame holding location-osm_id assignments.
    """
    transition_df = pd.read_csv(path_to_transition_file)
    location_df = location_df_from_csv(path_to_location_file)
    return transition_df, location_df


def persist_mobility_data_frames(transition_df, location_df, path_to_transition_file, path_to_location_file):
    """
    Persists the transition and location-osm_id assignment data frames.

    Parameters
    ----------
    transition_df : pd.core.frame.DataFrame
        A data frame holding transition information.
    location_df : pd.core.frame.DataFrame
        A data frame holding location-osm_id assignments.
    path_to_transition_file : str
        Path to transition file including file name and file ending.
    path_to_location_file : str
        Path to location file including file name and file ending.
    """
    transition_df.to_csv(path_to_transition_file, sep=",", index=False)
    location_df.to_csv(path_to_location_file, sep=",", index=False)


# todo: replace with function from detour detection once ready
def get_shortest_route_details(route, ignore_warnings=True):
    """

    Parameters
    ----------
    route : Route
    ignore_warnings

    Returns
    -------

    """
    ors_url = 'http://localhost:8008/ors'
    client = openrouteservice.Client(base_url=ors_url)
    input_route_has_radians = route.get_coordinates_unit() == 'radians'
    if input_route_has_radians:
        route.to_degrees_()

    route_directions = None
    detail_values = {}
    default_avg_speed_kmh = 45
    try:
        route_directions = client.directions((route[0], route[1]), profile='driving-car', format='geojson')
        shortest_route = Route(route_directions['features'][0]['geometry']['coordinates'], coordinates_unit='degrees')
    except Exception:
        print(f'Route could not be calculated for {route}. Shortest path will equal input route. Distance and '
              f'duration will be based on the direct connection and an assumed speed of {default_avg_speed_kmh} km/h.')
        # do a deep copy, when bug in Route is fixed
        shortest_route = route

    if route_directions is not None:
        route_details = route_directions['features'][0]['properties']['summary']
        for detail_name in ['distance', 'duration']:
            detail_present = detail_name in route_details.keys()
            if not detail_present:
                if not ignore_warnings:
                    warnings.warn(detail_name + f' value not available for {route}.')
                if detail_name == 'distance':
                    detail_values[detail_name] = get_distance(route[0], route[1])
                    if not ignore_warnings:
                        warnings.warn('Distance will be calculated from the direct connection (as the crow flies).')
                else:
                    detail_values[detail_name] = detail_values['distance'] / (default_avg_speed_kmh * 1_000 / 3_600)
                    if not ignore_warnings:
                        warnings.warn(f'Duration will be calculated based on an assumed speed of '
                                      f'{default_avg_speed_kmh} km/h.')
            else:
                detail_values[detail_name] = route_details[detail_name]
    else:
        detail_values['distance'] = get_distance(route[0], route[1])
        detail_values['duration'] = detail_values['distance'] / (default_avg_speed_kmh * 1_000 / 3_600)
    if input_route_has_radians:
        route.to_radians_()
        shortest_route.to_radians_()
    return detail_values['distance'], detail_values['duration'], shortest_route
