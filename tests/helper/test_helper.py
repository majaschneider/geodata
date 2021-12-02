import unittest
from de4l_geodata.helper.helper import get_digits


class TestHelper(unittest.TestCase):
    def test_get_digits(self):
        value = 0.123456789
        self.assertEqual(0, get_digits(value, 0))
        self.assertEqual(1, get_digits(value, 1))
        self.assertEqual(12345, get_digits(value, 5))
        self.assertEqual(123456789, get_digits(value, 9))
        self.assertEqual(1234567890, get_digits(value, 10))
