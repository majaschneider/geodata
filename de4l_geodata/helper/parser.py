"""A collection of functions to parse geodata.
"""

import warnings


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