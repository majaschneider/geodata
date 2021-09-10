from geodata.point import Point


class PointT(Point):
    """A point specifying a geographical location and a timestamp."""

    def __init__(self, coordinates, timestamp, geo_reference_system="latlon"):
        super().__init__(coordinates, geo_reference_system)
        self.timestamp = timestamp
