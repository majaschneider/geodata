"""A collection of functions to parse geodata.
"""

import warnings

import pandas as pd


def route_str_to_list(route_str):
    """
    Converts a route from string to list format.

    Parameters
    ----------
    route_str : str
        A route in string format. Example: [[-8.58, 41.14], [-8.5, 41.1]]

    Returns
    -------
    route_list : List
        The route converted into list format.
    """
    route_str = route_str.replace("[[", "[").replace("]]", "]")
    return points_str_to_list(route_str)


def points_str_to_list(points_str):
    """
    Converts a collection of points from string to list format.

    Parameters
    ----------
    points_str : str
        A points list in string format. Example: [-8.58, 41.14], [-8.5, 41.1]

    Returns
    -------
    route_list : List
        The point collection converted into list format.
    """
    route_list = []
    try:
        route_str = points_str.replace(" ", "").replace("],[", "];[").split(";")
        for point in route_str:
            point = point.replace("[", "").replace("]", "").split(",")
            route_list.append([float(point[0]), float(point[1])])
    except Exception:
        warnings.warn("An error occurred during parsing of route from string to list. An empty list will be "
                      "returned.")
    return route_list


def timestamps_str_to_list(timestamps_str):
    """
    Converts a collection of pandas.Timestamps from string to list format.

    Parameters
    ----------
    timestamps_str : str
        A string containing a list of pd.Timestamp objects, e.g.
        "[Timestamp('2020-01-01 10:00:00'), Timestamp('2020-01-02 15:00:00')]".

    Returns
    -------
    timestamps_list : List
        A list of pandas.Timestamp objects converted from timestamp_str.
    """
    timestamps_split = timestamps_str.replace('[', '').replace('Timestamp(', '').replace(')', '').replace(']', '') \
                           .replace(', ', '')[1:-1].split("''")
    timestamps_list = [pd.Timestamp(split) for split in timestamps_split]
    return timestamps_list


def float_str_to_list(float_str):
    """
    Converts a collection of Float from string to list format.

    Parameters
    ----------
    float_str : str
        A string containing a list of float objects, e.g. '[1, 2.5, 0, 7]'.

    Returns
    -------
    float_list : List
        A list of float objects converted from float_str.
    """
    float_list = [float(s) for s in float_str.replace('[', '').replace(']', '').replace(' ', '').split(',')]
    return float_list
