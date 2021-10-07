"""Provides helper functions for test classes.
"""

import numpy as np


def get_digits(value, number_of_decimals):
    """
    Returns the whole-number digits of value and number_of_decimals values after the comma.

    Parameters
    ----------
    value : float
        The value to apply this method to.
    number_of_decimals : int
        The number of decimal values to return.

    Returns
    -------
    int
        The whole-number digits of 'value' and its decimal digits up to 'number_of_decimals'.
    """
    return np.floor(value * np.power(10, number_of_decimals))
