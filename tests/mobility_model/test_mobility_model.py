import os
import unittest

import pandas as pd
from geopy.geocoders import Nominatim

from geodata.mobility_model import mobility_model
from geodata.geodatasets.taxi import TaxiServiceTrajectoryDataset as Td
from geodata.helper import parser


# todo: test all mobility model methods, when ors and nominatim are available from GitLab
class TestHelper(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_model_db = 'tests/mobility_model/test-taxi-model.db'
        try:
            os.remove(self.path_to_model_db)
        except Exception:
            pass
        scheme = 'http'
        ors_path = '172.17.2.117:50003'
        ors_profile = 'driving-car'
        nominatim_path = '172.17.2.117:50002'
        nominatim = Nominatim(scheme=scheme, domain=nominatim_path)

        # create model
        path_to_porto_taxi_file = 'tests/resources/test-taxi-dataset-profile.csv'
        self.model = mobility_model.MobilityModel(self.path_to_model_db, nominatim, ors_path, scheme, ors_profile)
        taxi_dataset = Td.create_from_csv(path_to_porto_taxi_file, max_allowed_speed_kmh=120)
        self.model.calculate_mobility_model(taxi_dataset.data_frame)

    def tearDown(self) -> None:
        try:
            os.remove(self.path_to_model_db)
        except Exception:
            pass

    def test_mobility_model(self):
        # expected: 5 transitions and 4 locations in mobility model
        # [-8.61004, 41.15299], [-8.61004, 41.1525] -> osm_id 1 -> 1
        # [-8.61004, 41.1525], [-8.610009757321686, 41.152922517153755]-> osm_id 1 -> 1
        # [-8.61004, 41.1525], [-8.61188, 41.15032]-> osm_id 1 -> 2
        # [-8.61188, 41.15032], [-8.61206, 41.15208]-> osm_id 2 -> 3
        # [-8.61206, 41.15208], [-8.6148, 41.15278]-> osm_id 3 -> 4

        expected_locations = [[[-8.61004, 41.15299], [-8.610009757321686, 41.152922517153755], [-8.61004, 41.1525]],
                              [[-8.61188, 41.15032]], [[-8.61206, 41.15208]], [[-8.6148, 41.15278]]]
        locations = pd.read_sql('''SELECT * FROM location;''', self.model.db)
        actual_locations = [parser.route_str_to_list(points) for points in locations['locations']]
        expected_locations.sort()
        actual_locations.sort()
        self.assertEqual(expected_locations, actual_locations)

        expected_transitions_cnt = [1, 1, 1, 2]
        actual_transitions_cnt = \
            pd.read_sql('''SELECT counter FROM transition;''', self.model.db)['counter'].to_list()
        actual_transitions_cnt.sort()
        self.assertEqual(expected_transitions_cnt, actual_transitions_cnt)

        expected_transitions_probabilities = [1 / 3, 2 / 3, 1, 1]
        actual_transition_probabilities = \
            pd.read_sql('''SELECT * FROM transition_probability;''', self.model.db)['transition_probability'].to_list()
        actual_transition_probabilities.sort()
        self.assertEqual(expected_transitions_probabilities, actual_transition_probabilities)
