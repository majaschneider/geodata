import random
import unittest
import torch
from geodata.route import Route
from geodata.point import Point
from geodata.point_t import PointT


class TestPointMethods(unittest.TestCase):
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
        r = Route([[0, 0]])
        self.assertEqual(Point, type(r[0]))

    def test_from_torch_tensor(self):
        tensor = torch.tensor([[0., 0.], [1., 1.]], requires_grad=True)
        r = Route([[0, 0], [1, 1]])
        r_from_tensor = Route.from_torch_tensor(tensor)
        self.assertEqual(r, r_from_tensor)

    def test_append(self):
        r = Route()
        r.append([0, 0])
        self.assertEqual(Route([[0, 0]]), r)

    def test_scale(self):
        scale_values = (-1, 1, -1, 1)
        r = Route([[-1, 1]])
        r.scale(scale_values)
        self.assertEqual(0, r[0].x_lon)
        self.assertEqual(1, r[0].y_lat)

    def test_inverse_scale(self):
        scale_values = (-1, 1, -1, 1)
        r = Route([[-1, 1]])
        r.scale(scale_values)
        r.inverse_scale(scale_values)
        self.assertEqual(-1, r[0].x_lon)
        self.assertEqual(1, r[0].y_lat)

    def test_pad(self):
        r = Route([[1, 1]])
        original_len = len(r)
        target_len = 3
        r.pad(target_len)
        self.assertEqual(target_len, len(r))
        for p in r[original_len:]:
            self.assertEqual(Point, type(p))
            self.assertEqual(0, p.x_lon)
            self.assertEqual(0, p.y_lat)

    def test_sort_by_time(self):
        r = Route()
        for i in range(100):
            r.append(PointT([1, 1], timestamp=random.randint(0, 1_000)))
        r.sort_by_time()
        timestamps = [p.timestamp for p in r]
        self.assertTrue(timestamps == sorted(timestamps))

    def test_has_timestamps(self):
        route_without_timestamps = Route([Point([0, 0])])
        self.assertFalse(route_without_timestamps.has_timestamps())
        route_with_timestamps = Route([PointT([0, 0], timestamp=0)])
        self.assertTrue(route_with_timestamps.has_timestamps())


if __name__ == "__main__":
    unittest.main()
