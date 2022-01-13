#!/usr/bin/env python
# vim:et:sta:sts=4:sw=4:ts=8:tw=79:

from __future__ import division
import numpy as np
from sympy import simplify
from application.functionplot.Function import Function
from application.functionplot.PointOfInterest import PointOfInterest as POI
from application.functionplot.helpers import fsolve, rfc, remove_outliers, BreakLoop, keep10
from logging import debug


class FunctionGraph:

    def get_limits(self):
        return ((self.x_min, self.x_max), (self.y_min, self.y_max))

    def zoom_default(self):
        self.auto = True
        self.update_graph_points()

    def zoom_x_in(self):
        self.auto = False
        self._zoom(zoom_out=False, zoom_x=True, zoom_y=False)

    def zoom_x_out(self):
        self.auto = False
        self._zoom(zoom_out=True, zoom_x=True, zoom_y=False)

    def zoom_y_in(self):
        self.auto = False
        self._zoom(zoom_out=False, zoom_x=False, zoom_y=True)

    def zoom_y_out(self):
        self.auto = False
        self._zoom(zoom_out=True, zoom_x=False, zoom_y=True)

    def zoom_in(self):
        self.auto = False
        self._zoom(zoom_out=False)

    def zoom_out(self):
        self.auto = False
        self._zoom(zoom_out=True)

    def _zoom(self, zoom_out=False, zoom_x=True, zoom_y=True,
              multiplier=1):
        if zoom_out:
            sf = self.scale_factor * multiplier
        else:
            sf = 1.0 / (self.scale_factor * multiplier)
        if zoom_x:
            x_center = (self.x_max + self.x_min) / 2
            x_range = self.x_max - self.x_min
            new_x_range = x_range * sf
            self.x_min = x_center - new_x_range / 2
            self.x_max = x_center + new_x_range / 2
        if zoom_y:
            y_center = (self.y_max + self.y_min) / 2
            y_range = self.y_max - self.y_min
            new_y_range = y_range * sf
            self.y_min = y_center - new_y_range / 2
            self.y_max = y_center + new_y_range / 2
        if self.logscale:
            if self.x_min < 0:
                self.x_min = 0
            if self.y_min < 0:
                self.y_min = 0
        self.update_graph_points()

    def update_graph_points(self):
        for f in self.functions:
            if f.visible:
                f.update_function_points([self.x_min, self.x_max,
                                       self.y_min, self.y_max])

    def add_function(self, expr, calc_poi):
        debug('Adding function: ' + expr)
        xylimits = [self.x_min, self.x_max, self.y_min, self.y_max]
        f = Function(expr, xylimits, calc_poi, self.logscale)
        if f.valid:
            self.functions.append(f)
            self.calc_intersections()
            self.update_xylimits()
            return True
        else:
            return False

    def update_xylimits(self):
        if self.auto:
            vertical_asymptotes = False
            horizontal_asymptotes = False
            debug('Calculating xylimits.')
            xl = []
            yl = []
            points = []
            # add function specific POIs
            for f in self.functions:
                if f.visible:
                    for p in f.poi:
                        if self.point_type_enabled[p.point_type]:
                            # don't add vertical or horizontal
                            # asymptotes here
                            if p.point_type < 6 or p.point_type > 7:
                                point = [p.x, p.y]
                                if point not in points:
                                    points.append([p.x, p.y])
            # add graph POIs (function intersections)
            for p in self.poi:
                if p.function[0].visible and p.function[1].visible \
                        and self.point_type_enabled[p.point_type]:
                    point = [p.x, p.y]
                    if point not in points:
                        points.append([p.x, p.y])
            # asymptotes
            # we need a trick to put asymptotes far away, but also
            # show them on the x axis. So, if there are any
            # asymptotes, we increase the size of the respective
            # axis 2 times.
            for f in self.functions:
                if f.visible:
                    for p in f.poi:
                        # vertical asymptotes
                        if p.point_type == 6:
                            if self.point_type_enabled[p.point_type]:
                                vertical_asymptotes = True
                                point = [p.x, 0]
                                if point not in points:
                                    points.append([p.x, 0])
                        # horizontal asymptotes
                        elif p.point_type == 7:
                            if self.point_type_enabled[p.point_type]:
                                horizontal_asymptotes = True
                                point = [0, p.y]
                                if point not in points:
                                    points.append([p.x, 0])
            # add default POIs (origin (0,0) etc)
            # only if other POIs are less than 3
            if len(points) < 3:
                for p in self.poi_defaults:
                    if self.point_type_enabled[p.point_type]:
                        point = [p.x, p.y]
                        if point not in points:
                            points.append([p.x, p.y])
            # gather everything together
            for point in points:
                xl.append(point[0])
                yl.append(point[1])
            # remove outliers
            if not self.outliers:
                # we need at least 9 points to detect outliers
                if len(xl) > 8:
                    debug('Trying to find outliers in X axis.')
                    xl = remove_outliers(xl)
                    debug('Trying to find outliers in Y axis.')
                    yl = remove_outliers(yl)
            x_min = min(xl)
            x_max = max(xl)
            x_range = x_max - x_min
            y_min = min(yl)
            y_max = max(yl)
            y_range = y_max - y_min
            # take care of edge cases, where all poi in an axis have
            # the same coordinate.
            if x_min == x_max:
                x_min = x_min - 2
                x_max = x_min + 2
                x_range = x_max - x_min
            if y_min == y_max:
                y_min = y_min - 2
                y_max = y_min + 2
                y_range = y_max - y_min
            # asymptotes. Increase the axis size in case any are
            # found
            if vertical_asymptotes:
                y_min = y_min - y_range
                y_max = y_max + y_range
                y_range = y_max - y_min
            if horizontal_asymptotes:
                x_min = x_min - x_range
                x_max = x_max + x_range
                x_range = x_max - x_min
            # find the max period of all functions involved and check
            # if at least 2 periods are shown
            periods = []
            for f in self.functions:
                if f.periodic:
                    periods.append(f.period)
            if len(periods) > 0:
                max_period = float(max(periods))
                x_middle = (x_max - x_min) / 2
                if x_range < 2 * max_period:
                    x_min = float(x_middle - 1.2 * max_period)
                    x_max = float(x_middle + 1.2 * max_period)
                    x_range = x_max - x_min
            # for some weird reason, setting all limits to integers
            # slightly breaks the sampling algorithm. Shift them by
            # a bit and everything works again. WIthout this
            # f(x)=sin(1/x) is not plotted properly for example.
            try:
                if x_min == int(x_min) and x_max == int(x_max):
                    x_min = x_min - 0.01 * x_range
                    x_max = x_max + 0.01 * x_range
                    x_range = x_max - x_min
                if y_min == int(y_min) and y_max == int(y_max):
                    y_min = y_min - 0.01 * y_range
                    y_max = y_max + 0.01 * y_range
                    y_range = y_max - y_min
            except OverflowError:
                pass
            debug('Setting X limits to ' +
                  str(x_min) + ' and ' + str(x_max))
            debug('Setting Y limits to ' +
                  str(y_min) + ' and ' + str(y_max))
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max
            # zoom out twice, gives better output
            self._zoom(zoom_out=True, zoom_x=True, zoom_y=True,
                       multiplier=2)
        if self.logscale:
            if self.x_min < 0:
                self.x_min = 0
                self.x_max = 100 * self.x_max
            if self.y_min < 0:
                self.y_min = 0
                self.y_max = 100 * self.y_max

    def grouped_poi(self, points):
        l = len(points)
        if l < 50:
            # max distance for grouped points is graph diagonal size /100
            x_range = self.x_max - self.x_min
            y_range = self.y_max - self.y_min
            # temp list of grouped points. Every group is a sublist
            c = []
            for p in points:
                # for intersection points, check for grouping only
                # if their respective functions are both visible
                # and only if the point type is enabled
                if p.point_type == 1 and p.function[0].visible \
                        and p.function[1].visible \
                        and self.point_type_enabled[p.point_type]:
                    c.append([p])
                # for non-intersection points, check for grouping
                # only if the respective function is visible and
                # only if the point type is enabled
                elif p.point_type != 1 and p.function.visible \
                        and self.point_type_enabled[p.point_type]:
                    c.append([p])
            done = False
            while not done:
                try:
                    l = len(c)
                    for i in range(0, l - 1):
                        for j in range(1, l):
                            if i != j:
                                for m in c[i]:
                                    for n in c[j]:
                                        if abs(m.x - n.x) < x_range / 100 and \
                                                abs(m.y - n.y) < y_range / 100:
                                            for k in c[j]:
                                                c[i].append(k)
                                            c.pop(j)
                                            raise BreakLoop
                    done = True
                except BreakLoop:
                    pass
            # final list of grouped points. For groups, return a single
            # point with coordinates the mean values of the coordinates
            # of the points that are grouped
            grouped = []
            for i in c:
                l = len(i)
                if l == 1:
                    grouped.append(i[0])
                else:
                    x_sum = 0
                    y_sum = 0
                    for j in i:
                        x_sum += j.x
                        y_sum += j.y
                    x = x_sum / l
                    y = y_sum / l
                    grouped.append(POI(x, y, 9, size=l))
        else:
            debug('Too many POI (' + str(l) + '). Disabling grouping.')
            grouped = points
        return grouped

    def calc_intersections(self):
        self.poi = []
        l = len(self.functions)
        for i in range(0, l - 1):
            f = self.functions[i]
            for j in range(i + 1, l):
                g = self.functions[j]
                self._calc_intersections_functions(f, g)

    # calculates the intersections between two functions
    def _calc_intersections_functions(self, f, g):
        debug('Looking for intersections between "' + f.expr +
              '" and "' + g.expr + '".')
        stored = False
        for i in self.intersections:
            f1 = i[0]
            f2 = i[1]
            px = i[2]
            py = i[3]
            if (f1 == f.simp_expr and f2 == g.simp_expr) or \
                    (f1 == g.simp_expr and f2 == f.simp_expr):
                p = POI(px, py, 1, function=[f, g])
                self.poi.append(p)
                stored = True
                debug('Stored intersection point: (' +
                      str(px) + ',' + str(py) + ')')
        if not stored:
            # FIXME: maybe I can do away with simplify here?
            d = str(f.simp_expr) + '-(' + str(g.simp_expr) + ')'
            try:
                ds = simplify(d)
                x = fsolve(ds)
                if x is None:
                    dnp = str(f.np_expr) + '-(' + str(g.np_expr) + ')'
                    x = self._calc_intersections_manually(dnp)
                for i in x:
                    y = f.simp_expr.subs('x', i)
                    xc = rfc(i)
                    yc = rfc(y)
                    if xc is not None and yc is not None:
                        p = POI(xc, yc, 1, function=[f, g])
                        self.poi.append(p)
                        self.intersections.append([f.simp_expr,
                                                   g.simp_expr, xc, yc])
                        debug('New intersection point: (' +
                              str(xc) + ',' + str(yc) + ')')
            except ValueError:
                debug('ValueError exception. Probably a ' +
                      'bug in sympy.')

    def _calc_intersections_manually(self, npexpr):
        debug('Calculating intersections manually')
        x = np.linspace(-20, 20, 10000)
        y = eval(npexpr)
        sol = []
        for i in range(2, len(y) - 1):
            if ((y[i] == 0) or
                    (y[i - 2] < y[i - 1] < 0 and y[i + 1] > y[i] > 0) or
                    (y[i - 2] > y[i - 1] > 0 and y[i + 1] < y[i] < 0)):
                sol.append(x[i])
        sol = keep10(sol)
        return sol

    def clear(self):
        self.x_min = -1.2
        self.x_max = 1.2
        self.y_min = -1.2
        self.y_max = 1.2
        self.logscale = False
        self.auto = True

    def new(self):
        self.visible = True
        self.show_legend = True
        self.legend_location = 1  # upper right
        self.show_poi = True

        self.outliers = False
        self.grouped = True

        self.functions = []
        self.poi = []
        self.poi_defaults = []
        self.poi_defaults.append(POI(0, 0, 0))
        self.poi_defaults.append(POI(0, 1, 0))
        self.poi_defaults.append(POI(0, -1, 0))
        self.poi_defaults.append(POI(1, 0, 0))
        self.poi_defaults.append(POI(-1, 0, 0))
        self.clear()

    def __init__(self):
        self.scale_factor = 1.2
        # we have 8 types of POIs
        self.point_type_enabled = [
            True,  # 0: standard axis points
            True,  # 1: function intersections
            True,  # 2: x intercepts
            True,  # 3: y intercept
            True,  # 4: local min/max
            True,  # 5: inflection points
            True,  # 6: vertical asymptotes
            True,  # 7: horizontal asymptotes
            True,  # 8: slope is 45 or -45 degrees
            True  # 9: grouped POIs
        ]
        self.intersections = []
        self.new()
