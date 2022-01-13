#!/usr/bin/env python
# vim:et:sta:sts=4:sw=4:ts=8:tw=79:


class PointOfInterest:
    """ A class used to represent a point of interest."""

    def __init__(self, x, y, point_type=None, size=1, function=None, color=None):
        """

        :param x: float
            x coordinate of the point of interest
        :param y: float
            y coordinate of the point of interest
        :param point_type: str
            The type of the point of interest. Can be one of the types bellow:
            # POI types:
            # 0: standard axis points {(0,0), (0,1), (1,0)}
            # 1: function intersections
            # 2: x intercept
            # 3: y intercept
            # 4: local min/max
            # 5: inflection points
            # 6: vertical asymptotes
            # 7: horizontal asymptotes
            # 8: slope is 45 or -45 degrees
            # 9: POI group
        :param size: int
            The size used for grouping points of interest that is close enough accordingly
            to the zoom scale. The size growing bigger as new points joining the specific
            group
        :param function: ?
            The function that the point of interest belong to
        :param color: ?
            The color the group of points of interest has
        """
        self.x = x
        self.y = y
        self.size = size
        self.function = function
        self.color = color
        self.point_type = point_type
