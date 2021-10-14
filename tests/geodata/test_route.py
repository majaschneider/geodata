import random
import unittest
import torch
from geodata.route import Route
from geodata.point import Point
from geodata.point_t import PointT


class TestPointMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.route_without_timestamps = Route([Point([0, 0])])
        self.route_with_timestamps = Route([PointT([0, 0], timestamp=0)])

    def test_constructor(self):
        # initializes with empty list
        self.assertEqual(Route([]), Route())
        with self.assertRaises(TypeError):
            # disallow other list items than lists
            Route(1)
            Route(1.0)
            Route("1")
            # disallow lists as list items that are not of type Point or can be converted into Point
            Route([[]])
            Route([[1]])
            Route([[1, 2, 3]])
        # successfully creates a Route
        self.assertEqual(Route, type(Route([[0, 0]])))
        self.assertEqual(Route, type(Route([[0, 0], [1, 1]])))
        self.assertEqual(Route, type(Route([Point([0, 0])])))
        self.assertEqual(Route, type(Route([Point([0, 0]), Point([1, 1])])))

    def test_set_item(self):
        # assures that list items are points
        route = Route([[0, 0]])
        self.assertEqual(Point, type(route[0]))

    def test_from_torch_tensor(self):
        tensor = torch.tensor([[0., 0.], [1., 1.]], requires_grad=True)
        route = Route([[0, 0], [1, 1]])
        route_from_tensor = Route.from_torch_tensor(tensor)
        self.assertEqual(route, route_from_tensor)

    def test_append(self):
        route = Route()
        route.append([0, 0])
        self.assertEqual(Route([[0, 0]]), route)

    def test_scale(self):
        scale_values = (-1, 1, -1, 1)
        r = Route([[-1, 1]])
        r.scale(scale_values)
        self.assertEqual(0, r[0].x_lon)
        self.assertEqual(1, r[0].y_lat)

    def test_inverse_scale(self):
        scale_values = (-1, 1, -1, 1)
        route = Route([[-1, 1]])
        route.scale(scale_values)
        route.inverse_scale(scale_values)
        self.assertEqual(-1, route[0].x_lon)
        self.assertEqual(1, route[0].y_lat)

    def test_pad(self):
        # method only applicable for routes containing items of type Point, but no subclasses of Point
        with self.assertRaises(Exception):
            self.route_with_timestamps.pad(5)
        try:
            self.route_without_timestamps.pad(5)
        except Exception:
            self.fail("Unexpected exception when invoking pad().")

        route = Route([[1, 1]])
        original_len = len(route)
        target_len = 3
        route.pad(target_len)
        self.assertEqual(target_len, len(route))
        for point in route[original_len:]:
            self.assertEqual(Point, type(point))
            self.assertEqual(0, point.x_lon)
            self.assertEqual(0, point.y_lat)

    def test_sort_by_time(self):
        # method only applicable for routes containing items of type PointT, but not Point
        with self.assertRaises(Exception):
            self.route_without_timestamps.sort_by_time()
        try:
            self.route_with_timestamps.sort_by_time()
        except Exception:
            self.fail("Unexpected exception when invoking sort_by_time().")

        route = Route()
        for i in range(100):
            route.append(PointT([1, 1], timestamp=random.randint(0, 1_000)))
        route.sort_by_time()
        timestamps = [p.timestamp for p in route]
        self.assertTrue(timestamps == sorted(timestamps))

    def test_has_timestamps(self):
        self.assertFalse(self.route_without_timestamps.has_timestamps())
        self.assertTrue(self.route_with_timestamps.has_timestamps())


if __name__ == "__main__":
    unittest.main()
