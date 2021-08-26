"""Provides helper functions for test classes."""

import numpy as np


def get_digits(value, number_of_decimals):
    """
    Returns the whole-number digits of value and number_of_decimals values after the comma.
    :param value: value to apply this method to
    :param number_of_decimals: number of decimal values to return
    :return: whole-number digits of value and decimal digits up to number_of_decimals
    """
    return np.floor(value * np.power(10, number_of_decimals))
