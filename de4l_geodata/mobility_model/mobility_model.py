import sqlite3

import pandas as pd
from geopy.geocoders import Nominatim
from de4l_detour_detection.detour_detection import get_directions_for_route

from de4l_geodata.geodata.route import Route
from de4l_geodata.geodata.point import Point
from de4l_geodata.helper import parser


def execute_sql(sql, db, data=None):
    cursor = db.cursor()
    if data is None:
        cursor.execute(sql)
    else:
        cursor.execute(sql, data)
    db.commit()
    return cursor.lastrowid


class MobilityModel:
    def __init__(self, path_to_file, nominatim, ors_path, ors_profile):
        """
        Initializes this mobility model.

        Parameters
        ----------
        path_to_file : str
            The path pointing to the database file that will hold the mobility information of this model, including
            file ending '.db'. The path to the folder of this file needs to exist.
        nominatim : Nominatim
            A running instance of Nominatim.
        ors_path : str
            The base address of an available instance of Openrouteservice. Its structure should be '[host]:[port]'.
        ors_profile :
            {'driving-car', 'driving-hgv', 'foot-walking', 'foot-hiking', 'cycling-regular',
            'cycling-road', 'cycling-mountain', 'cycling-electric'}
            Specifies the mode of transport to use when calculating directions.
            See: https://openrouteservice-py.readthedocs.io/en/latest/#module-openrouteservice.directions
        """
        self.nominatim = nominatim

        self.ors_path = ors_path
        self.ors_profile = ors_profile

        # create databases or establish connection
        self.db = sqlite3.connect(path_to_file)

        # create database tables if they don't exist
        self.create_location_table()
        self.create_transition_table()

        self.nr_routes = 0

    def calculate_mobility_model(self, df):
        """
        Calculates transition probabilities between street segments per weekday and the assignment of locations
        to openstreetmap segment ids.

        Parameters
        ----------
        df : pd.core.frame.DataFrame
            A data frame holding mobility data, which contains the following columns:
            trip_time_start_utc : pd.Timestamp
                The start timestamp of a mobility trace.
            route : Route
                The mobility trace in 'radians' format.
        """

        df = df.copy()
        df['weekday'] = df['trip_time_start_utc'].dt.weekday

        for _, row in df.iterrows():
            # assume 'radians' coordinates
            route = Route(row['route'])
            weekday = row['weekday']
            route_len = len(route)
            self.nr_routes += 1

            for i in range(route_len - 1):
                current_point = route[i].to_degrees(ignore_warning=True)
                next_point = route[i + 1].to_degrees(ignore_warning=True)

                current_osm_id = self.nominatim.reverse([current_point.y_lat, current_point.x_lon]).raw['osm_id']
                next_osm_id = self.nominatim.reverse([next_point.y_lat, next_point.x_lon]).raw['osm_id']

                directions = \
                    get_directions_for_route(Route([current_point, next_point]), self.ors_path, self.ors_profile)
                distance = directions[0]['distance']
                duration = directions[0]['duration']

                self.create_or_update_transition(current_osm_id, next_osm_id, weekday, distance, duration)

                # create or update the location of the current (and maybe next) segment
                segments_to_check = [(current_osm_id, current_point)]
                if i == route_len - 2:
                    segments_to_check.append((next_osm_id, next_point))
                for osm_id, point in segments_to_check:
                    self.create_or_update_location(osm_id, point)
        self.calculate_transition_probabilities()

    def calculate_transition_probabilities(self):
        """
        Calculates transition probabilities based on the transition table. If existing, the transition probability table
        will be dropped and overwritten.
        """
        execute_sql('''DROP TABLE IF EXISTS transition_probability;''', self.db)
        self.create_transition_probability_table()
        sql = '''INSERT INTO transition_probability
                 SELECT transition_id, CAST(counter AS float) / CAST(sum_of_counter AS float) AS transition_probability
                 FROM transition LEFT JOIN
                    (SELECT osm_id_1, weekday, SUM(counter) AS sum_of_counter
                    FROM transition GROUP BY osm_id_1, weekday) AS counter_stats
                 ON transition.osm_id_1 = counter_stats.osm_id_1 AND transition.weekday = counter_stats.weekday;'''
        execute_sql(sql, self.db)
        
    def create_transition(self, new_transition):
        """
        Creates a new transition with the given parameters osm_id_1, osm_id_2, weekday, counter, distance_sum and
        duration_sum.

        Parameters
        ----------
        new_transition : tuple
            A transition tuple containing osm_id_1, osm_id_2, weekday, counter, distance_sum and duration_sum of the
            transition.
        """
        sql = '''INSERT INTO transition(osm_id_1,osm_id_2,weekday,counter,distance_sum,duration_sum)
                 VALUES(?,?,?,?,?,?);'''
        execute_sql(sql, self.db, new_transition)

    def update_transition(self, transition_update):
        """
        Updates the transition table with the parameters counter, distance_sum, duration_sum, osm_id_1, osm_id_2 and
        weekday.

        Parameters
        ----------
        transition_update : tuple
            A transition tuple containing counter, distance_sum, duration_sum, osm_id_1, osm_id_2 and weekday of the
            transition update.
        """
        sql = '''UPDATE transition set counter=?, distance_sum=?, duration_sum=?
                 WHERE osm_id_1=? AND osm_id_2=? AND weekday=?;'''
        execute_sql(sql, self.db, transition_update)

    def create_or_update_transition(self, osm_id_1, osm_id_2, weekday, distance, duration):
        """
        Creates a transition with the given parameters, if a transition between current_osm_id and next_osm_id is not
        existing in the transition table. Otherwise, updates the found transition with the given values.

        Parameters
        ----------
        osm_id_1 : int
            The openstreetmap segment id of the start segment of the transition.
        osm_id_2 : int
            The openstreetmap segment id of the destination segment of the transition.
        weekday : range(7)
            The weekday of the transition.
        distance : float
            The distance between start and destination segment.
        duration : int
            The duration of travel in seconds between start and destination segment.
        """
        existing_transition = self.find_transition(osm_id_1, osm_id_2, weekday)
        if existing_transition.empty:
            self.create_transition(
                (osm_id_1, osm_id_2, weekday, 1, distance, duration)
            )
        else:
            self.update_transition(
                (int(existing_transition['counter'].values[0] + 1),
                 existing_transition['distance_sum'].values[0] + distance,
                 existing_transition['duration_sum'].values[0] + duration, osm_id_1, osm_id_2, weekday)
            )

    def create_location(self, new_location):
        """
        Creates a location in the location table with the given parameters osm_id, locations and center.

        Parameters
        ----------
        new_location : tuple
            A location tuple, containing osm_id, seen locations in this segment and the center of the locations.
        """
        sql = '''INSERT INTO location(osm_id,locations,center)
                 VALUES(?,?,?);'''
        execute_sql(sql, self.db, new_location)
        
    def update_location(self, location_update):
        """
        Updates the location table with the given parameters locations, center and osm_id.
    
        Parameters
        ----------
        location_update : tuple
            A location tuple containing seen locations in the segment of the location, the center of the locations and
            the osm_id of the location.
        """
        sql = '''UPDATE location set locations=?, center=?
                 WHERE osm_id=?;'''
        execute_sql(sql, self.db, location_update)

    def create_or_update_location(self, osm_id, point):
        """
        Creates a location with the given parameters, if a location for osm_id is not existing in the location table.
        Otherwise, updates the found location with the given values.


        Parameters
        ----------
        osm_id : int
            The openstreetmap segment id of the location.
        point : Point
            The point belonging to the osm_id.
        """
        existing_location = self.find_location(osm_id)
        if existing_location.empty:
            self.create_location(
                (osm_id, str([point]), str(point))
            )
        else:
            # setting 'degrees' ignores a potential violation of the value range in case coordinates_unit is 'radians'
            locations = Route(existing_location['locations'].values[0], coordinates_unit='degrees')
            if point not in locations:
                center = locations.append(point).get_average_point()
                self.update_location(
                    (str(locations), str(center), osm_id)
                )
    
    def find_transition(self, osm_id_1, osm_id_2, weekday):
        """
        Finds the rows describing the transition from osm_id_1 to osm_id_2 on a certain weekday.

        Parameters
        ----------
        osm_id_1 : int
            The id of the openstreetmap road segment of the transition start.
        osm_id_2 : int
            The id of the openstreetmap road segment of the transition end.
        weekday : int
            The weekday of the transition.

        Returns
        -------
        pd.core.frame.DataFrame
            The data frame describing the transition matching the given parameters, containing transition_id, osm_id_1,
            osm_id_2, weekday, counter, distance_sum, duration_sum, distance_avg, duration_avg. Will be empty, if no
            transition found.
        """
        sql = f"SELECT * FROM transition WHERE osm_id_1={osm_id_1} AND osm_id_2={osm_id_2} AND weekday={weekday};"
        return pd.read_sql(sql, self.db)

    def find_possible_transition(self, start_osm_id, forbidden_target_osm_ids, weekday):
        """
        Finds the rows describing all possible transitions starting from start_osm_id on the given weekday and ending
        in a segment, which is not contained in forbidden_target_osm_ids.

        Parameters
        ----------
        start_osm_id : int
            The id of the openstreetmap road segment of the transition start.
        forbidden_target_osm_ids : List[int]
            The ids of the openstreetmap road segments that are forbidden as target of the transition.
        weekday : int
            The weekday of the transition.

        Returns
        -------
        pd.core.frame.DataFrame
            The data frame describing the transition matching the given constraints, containing transition_id, osm_id_1,
            osm_id_2, weekday, counter, distance_sum, duration_sum, distance_avg, duration_avg and
            transition_probability. Will be empty, if no transition is found.
        """
        forbidden_target_osm_ids_str = str(forbidden_target_osm_ids).replace('[', '').replace(']', '')
        sql = f"SELECT t.*, p.transition_probability " \
              f"FROM transition t LEFT JOIN transition_probability p " \
              f"ON t.transition_id=p.transition_id " \
              f"WHERE osm_id_1={start_osm_id} AND " \
              f"osm_id_2 NOT IN ({forbidden_target_osm_ids_str}) AND weekday={weekday};"
        return pd.read_sql(sql, self.db)

    def find_location(self, osm_id):
        """
        Finds the rows describing the location assigned to osm_id.

        Parameters
        ----------
        osm_id : int
            The id of the searched openstreetmap road segment.

        Returns
        -------
        pd.core.frame.DataFrame
            The data frame describing the location with the given osm_id, containing osm_id, locations, center. Will be
            empty, if no location found.
        """
        sql = f"SELECT * FROM location WHERE osm_id={osm_id};"
        df = pd.read_sql(sql, self.db)
        df['locations'] = df['locations'].apply(
            lambda locations: Route(parser.route_str_to_list(locations), coordinates_unit='degrees')
        )
        df['center'] = df['center'].apply(
            lambda center: Point(parser.points_str_to_list(center)[0], coordinates_unit='degrees')
        )
        return df

    def create_transition_table(self):
        """Creates transition table.
        """
        sql = '''CREATE TABLE IF NOT EXISTS transition
                 (transition_id INTEGER PRIMARY KEY,
                  osm_id_1 INTEGER, 
                  osm_id_2 INTEGER, 
                  weekday INTEGER, 
                  counter INTEGER, 
                  distance_sum REAL,
                  duration_sum REAL,
                  distance_avg REAL GENERATED ALWAYS AS (distance_sum / counter) STORED, 
                  duration_avg REAL GENERATED ALWAYS AS (duration_sum / counter) STORED);'''
        execute_sql(sql, self.db)

    def create_transition_probability_table(self):
        """Creates transition probability table.
        """
        sql = '''CREATE TABLE IF NOT EXISTS transition_probability
                 (transition_id INTEGER PRIMARY KEY,
                  transition_probability REAL);'''
        execute_sql(sql, self.db)

    def create_location_table(self):
        """Creates location table.
        """
        sql = '''CREATE TABLE IF NOT EXISTS location
                 (osm_id INTEGER PRIMARY KEY, 
                  locations TEXT, 
                  center TEXT);'''
        execute_sql(sql, self.db)
