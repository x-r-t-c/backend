#!/usr/bin/env python
# vim:et:sta:sts=4:sw=4:ts=8:tw=79:

from __future__ import division
import numpy as np
from sympy import diff, limit, simplify, latex, pi
# from sympy.functions import Abs
from application.functionplot.PointOfInterest import PointOfInterest as POI
from application.functionplot.helpers import pod, fsolve, rfc, sample, keep10  # log10,
from logging import debug
import re


win32 = True
try:
    import winshell
except ImportError:
    win32 = False

# basicConfig(level=DEBUG)

# don't use multiprocessing on win32. It causes huge slowdowns
# because of the lack of os.fork()
if win32:
    from threading import Thread as fProcess
    from queue import Queue as fQueue
else:
    from multiprocessing import Process as fProcess
    from multiprocessing import Queue as fQueue


class Function:

    def update_function_points(self, xylimits):
        x_min, x_max, y_min, y_max = xylimits
        x_initial = np.linspace(x_min, x_max, self.resolution)
        try:
            x, y_arr = sample(self.np_expr, x_initial)
            y = y_arr[0]
        except IndexError:
            # if f(x)=a, make sure that y is an array with the
            # same size as x and with a constant value.
            debug('This looks like a constant function: ' +
                  self.np_expr)
            self.constant = True
            # 2 graph points are enough for a constant function
            x = np.array([x_min, x_max])
            this_y = eval(self.np_expr)
            y = np.array([this_y, this_y])
        except Exception as e:
            debug('Exception caught.' +
                  'This should not have happened here: ' + str(e))
            return False
        # no need to calculate values that are off the displayed
        # scale. This fixes some trouble with asymptotes like in
        # tan(x).
        if not self.constant:
            y[y > y_max] = np.inf
            y[y < y_min] = -np.inf
        y_mean = (y_max + y_min) / 2
        # delete consecutive inf or -inf values
        delete = []
        for i in range(1, len(x)):
            # print(f'y[{i}]= {y[i]}\n')
            if y[i] == np.inf and y[i - 1] == np.inf:
                delete.append(i)
            if y[i] == -np.inf and y[i - 1] == -np.inf:
                delete.append(i)
        x = np.delete(x, delete)
        y = np.delete(y, delete)
        # add points at the edge of the graph
        addyinf = []
        addx = []
        for i in range(1, len(x)):
            if y[i] == np.inf and y[i - 1] > y_mean:
                addyinf.append(i)
                addx.append(x[i - 1])
            if y[i] > y_mean and y[i - 1] == np.inf:
                addyinf.append(i)
                addx.append(x[i])
        x = np.insert(x, addyinf, addx)
        y = np.insert(y, addyinf, y_max)
        addyinf = []
        addx = []
        for i in range(1, len(x)):
            if y[i] == -np.inf and y[i - 1] < y_mean:
                addyinf.append(i)
                addx.append(x[i - 1])
            if y[i] < y_mean and y[i - 1] == -np.inf:
                addyinf.append(i)
                addx.append(x[i])
        x = np.insert(x, addyinf, addx)
        y = np.insert(y, addyinf, y_min)

        self.graph_points = x, y
        debug('Number of points calculated:' + str(len(x)))
        return True

    def _get_expr(self, expr):
        # caps to lowercase
        expr = expr.lower()
        # remove all spaces
        expr = expr.replace(' ', '')
        # replace commas with decimals
        expr = expr.replace(',', '.')
        # braces and bracets to parens
        expr = expr.replace('{', '(')
        expr = expr.replace('}', ')')
        expr = expr.replace('[', '(')
        expr = expr.replace(']', ')')
        # exp contains an x that we don't like
        expr = expr.replace('exp(', 'ep(')
        # turn greek (unicode) notation to english
        expr = expr.replace('\xce\xb7\xce\xbc(', 'sin(')
        expr = expr.replace('\xcf\x83\xcf\x85\xce\xbd(', 'cos(')
        expr = expr.replace('\xce\xb5\xcf\x86(', 'tan(')
        expr = expr.replace('\xcf\x83\xcf\x86(', 'cot(')
        expr = expr.replace('\xcf\x83\xcf\x84\xce\xb5\xce\xbc(',
                            'csc(')
        expr = expr.replace('\xcf\x84\xce\xb5\xce\xbc(', 'sec(')
        expr = expr.replace('\xcf\x87', 'x')
        expr = expr.replace('\xcf\x80', 'pi')
        # implied multiplication
        expr = re.sub('([0-9])([a-z\(])', '\\1*\\2', expr)
        expr = re.sub('([a-z\)])([0-9])', '\\1*\\2', expr)
        expr = re.sub('(pi)([a-z\(])', '\\1*\\2', expr)
        expr = re.sub('([a-z\)])(pi)', '\\1*\\2', expr)
        expr = re.sub('(\))([a-z\(])', '\\1*\\2', expr)
        expr = re.sub('(x)([a-z\(])', '\\1*\\2', expr)
        # bring back exp
        expr = expr.replace('ep(', 'exp(')
        return expr

    def _get_np_expr(self, expr):
        # add "np." prefix to trig functions
        expr = expr.replace('sin(', 'np.sin(')
        expr = expr.replace('cos(', 'np.cos(')
        expr = expr.replace('tan(', 'np.tan(')
        expr = expr.replace('cot(', '1/np.tan(')  # no cot in numpy
        expr = expr.replace('sec(', '1/np.cos(')  # no sec,
        expr = expr.replace('csc(', '1/np.sin(')  # and no csc either
        # correct log functions
        expr = expr.replace('log10(', 'np.log10(')
        expr = expr.replace('log(', 'np.log(')
        # square root
        expr = expr.replace('sqrt(', 'np.sqrt(')
        # absolute value. For numpy, turn the sympy Abs function
        # to np.abs
        expr = expr.replace('Abs(', 'np.abs(')
        # pi and e (e also covers exp->np.exp)
        expr = expr.replace('pi', 'np.pi')
        expr = expr.replace('e', 'np.e')
        # powers
        expr = expr.replace('^', '**')
        return expr

    def _simplify_expr(self, expr):
        # sympy.functions.Abs is imported as Abs so we're using it
        # that way with sympy
        expr = expr.replace('abs(', 'Abs(')
        # we need to convert e to a float value. Since sec() is the
        # only function that also includes an "e", we'll remove
        # that temporarily
        #expr = expr.replace('sec(', 'scc(')
        #expr = expr.replace('e', '2.7183')
        #expr = expr.replace('scc(', 'sec(')
        # log() in sympy is natural log. log10() is defined
        # in helpers.py
        expr = expr.replace('log(', 'log10(')
        expr = expr.replace('ln(', 'log(')

        simp_expr = simplify(expr)
        debug('"' + expr + '" has been simplified to "' +
              str(simp_expr) + '"')
        return simp_expr

    def _get_mathtex_expr(self, expr):
        # expr is a simplified sympy expression. Creates a LaTeX
        # string from the expression using sympy latex printing.
        e = latex(expr)
        e = e.replace('log{', 'ln{')
        e = e.replace('log_{10}', 'log')
        e = e.replace('\\lvert', '|')
        e = e.replace('\\rvert', '|')
        e = '$' + e + '$'
        return e

    def _calc_y_intercept(self, q, expr):
        debug('Looking for the y intercept for: ' + str(expr))
        y = expr.subs('x', 0)
        if str(y) == 'zoo' or str(y) == 'nan':
            # 'zoo' is imaginary infinity
            debug('The Y axis is actually a vertical asymptote.')
            debug('Added vertical asymptote (0,0)')
            q.put(POI(0, 0, 6))
        else:
            yc = rfc(y)
            if yc is not None and 'inf' not in str(yc):
                debug('Added y intercept at (0,' + str(yc) + ')')
                q.put(POI(0, yc, 3))
            else:
                q.put(None)
        debug('Done calculating y intercept')

    def _calc_x_intercepts(self, q, expr):
        debug('Looking for x intercepts for: ' + str(expr))
        x = fsolve(expr)
        poi = []
        manual = False
        if x is None:
            manual = True
            x = self._calc_x_intercepts_manually()
        for xc in x:
            poi.append(POI(xc, 0, 2))
            debug('Added x intercept at (' + str(xc) + ',0)')
        # try to find if the function is periodic using the
        # distance between the x intercepts
        if self.trigonometric and not self.periodic and \
                not self.polynomial and not manual and x != []:
            debug('Checking if function is periodic using' +
                  ' x intercepts.')
            if not manual:
                self.check_periodic(x)
        if poi == []:
            debug('Done calculating x intercepts. None found.')
        else:
            debug('Done calculating x intercepts')
        q.put(poi)

    def _calc_x_intercepts_manually(self):
        debug('Calculating x intercepts manually')
        x = np.linspace(self.x_min_manual, self.x_max_manual, 10000)
        y = eval(self.np_expr)
        sol = []
        for i in range(2, len(y) - 1):
            if ((y[i] == 0) or
                    (y[i - 2] < y[i - 1] < 0 and y[i + 1] > y[i] > 0) or
                    (y[i - 2] > y[i - 1] > 0 and y[i + 1] < y[i] < 0)):
                sol.append(x[i])
        sol = keep10(sol)
        return sol

    def _calc_min_max(self, q, f1, expr):
        debug('Looking for local min/max for: ' + str(expr))
        x = fsolve(f1)
        poi = []
        manual = False
        if x is None:
            manual = True
            x = self._calc_min_max_manually()
        for xc in x:
            y = expr.subs('x', xc)
            yc = rfc(y)
            if yc is not None:
                poi.append(POI(xc, yc, 4))
                debug('Added local min/max at (' + str(xc) + ',' +
                      str(yc) + ')')
        if self.trigonometric and not self.periodic and \
                not self.polynomial and not manual and x != []:
            debug('Checking if function is periodic using' +
                  ' min/max.')
            if not manual:
                self.check_periodic(x)
        if poi == []:
            debug('Done calculating min/max. None found.')
        else:
            debug('Done calculating min/max')
        q.put(poi)

    def _calc_min_max_manually(self):
        debug('Calculating local min/max manually')
        x = np.linspace(self.x_min_manual, self.x_max_manual, 10000)
        y = eval(self.np_expr)
        sol = []
        for i in range(2, len(y) - 2):
            dx = x[i - 1] - x[i]
            dy = y[i - 1] - y[i]
            if y[i - 2] > y[i - 1] > y[i] < y[i + 1] < y[i + 2] and \
                    0 <= abs(dy / dx) < 10:
                sol.append(x[i])
            elif y[i - 2] < y[i - 1] < y[i] > y[i + 1] > y[i + 2] and\
                    0 <= abs(dy / dx) < 10:
                sol.append(x[i])
        sol = keep10(sol)
        return sol

    def _calc_inflection(self, q, f2, expr):
        debug('Looking for inflection points for: ' + str(expr))
        x = fsolve(f2)
        poi = []
        manual = False
        if x is None:
            manual = True
            x = self._calc_inflection_manually(f2)
        for xc in x:
            y = expr.subs('x', xc)
            yc = rfc(y)
            if yc is not None:
                poi.append(POI(xc, yc, 5))
                debug('Added inflection point at (' +
                      str(xc) + ',' + str(yc) + ')')
        if self.trigonometric and not self.periodic and \
                not self.polynomial and not manual and x != []:
            debug('Checking if function is periodic using' +
                  ' inflection points.')
            if not manual:
                self.check_periodic(x)
        if poi == []:
            debug('Done calculating inflection points. None found.')
        else:
            debug('Done calculating inflection points')
        q.put(poi)

    def _calc_inflection_manually(self, f2):
        debug('Calculating inflection points manually')
        npexpr = self._get_np_expr(str(f2))
        x = np.linspace(self.x_min_manual, self.x_max_manual, 10000)
        sol = []
        try:
            y = eval(npexpr)
            for i in range(2, len(y) - 1):
                if ((y[i] == 0) or
                        (y[i - 2] < y[i - 1] < 0 and y[i] > 0) or
                        (y[i - 2] > y[i - 1] > 0 and y[i] < 0)):
                    sol.append(x[i])
        except NameError:
            debug('Not possible to evaluate second derivative')
        except TypeError:
            debug('Second derivative is probably constant. ' +
                  'No inflection points found.')
        sol = keep10(sol)
        return sol

    def _calc_slope_45(self, q, f1, expr):
        debug('Looking for points where slope is 45 degrees for: ' +
              str(expr))
        x1 = fsolve(f1 - 1)
        x2 = fsolve(f1 + 1)
        poi = []
        manual = False
        if x1 is None and x2 is None:
            manual = True
            x = self._calc_slope45_manually(f1)
        elif x1 is None:
            x = x2
        elif x2 is None:
            x = x1
        else:
            x = x1 + x2
        for xc in x:
            y = expr.subs('x', xc)
            yc = rfc(y)
            if yc is not None:
                poi.append(POI(xc, yc, 8))
                debug('Added slope45 point at (' +
                      str(xc) + ',' + str(yc) + ')')
        if self.trigonometric and not self.periodic and \
                not self.polynomial and x != []:
            debug('Checking if function is periodic using' +
                  ' slope45 points.')
            if not manual:
                self.check_periodic(x)
        if poi == []:
            debug('Done calculating slope45 points. None found.')
        else:
            debug('Done calculating slope45 points')
        q.put(poi)

    def _calc_slope45_manually(self, f1):
        debug('Calculating slope45 points manually')
        npexpr = self._get_np_expr(str(f1))
        x = np.linspace(self.x_min_manual, self.x_max_manual, 10000)
        sol = []
        try:
            y = eval(npexpr)
            for i in range(1, len(y) - 1):
                if ((y[i] == 1) or (y[i] == -1) or
                        (y[i - 1] < 1 and y[i] > 1) or
                        (y[i - 1] > 1 and y[i] < 1) or
                        (y[i - 1] < -1 and y[i] > -1) or
                        (y[i - 1] > -1 and y[i] < -1)):
                    sol.append(x[i])
        except NameError:
            debug('Not possible to evaluate first derivative')
        except TypeError:
            debug('First derivative is probably constant. ' +
                  'No slope45 points found.')
        sol = keep10(sol)
        return sol

    def _calc_vertical_asym(self, q, expr):
        debug('Looking for vertical asymptotes for: ' + str(expr))
        x = pod(expr, 'x')
        poi = []
        for i in x:
            y = expr.subs('x', i)
            xc = rfc(i)
            # yc = float(y) # this returns inf.
            # we'll just put vertical asymptotes on the x axis
            if xc is not None:
                yc = 0
                poi.append(POI(xc, yc, 6))
                debug('Added vertical asymptote (' + str(xc) + ',' +
                      str(yc) + ')')
        if self.trigonometric and not self.periodic and \
                not self.polynomial and x != []:
            debug('Checking if function is periodic using' +
                  ' vertical asymptotes.')
            self.check_periodic(x)
        if poi == []:
            debug('Done calculating vertical asymptotes.' +
                  'None found.')
        else:
            debug('Done calculating vertical asymptotes')
        q.put(poi)

    def _calc_horizontal_asym(self, q, expr):
        # if the limit(x->+oo)=a, or limit(x->-oo)=a, then
        # y=a is a horizontal asymptote.
        debug('Looking for horizontal asymptotes for: ' +
              str(expr))
        try:
            poi = []
            lr = limit(expr, 'x', 'oo')
            ll = limit(expr, 'x', '-oo')
            if 'oo' not in str(lr):
                debug('Found a horizontal asymptote at y=' +
                      str(lr) + ' as x->+oo.')
                poi.append(POI(0, lr, 7))
            if 'oo' not in str(ll):
                if ll == lr:
                    debug('Same horizontal asymptote as x->-oo.')
                else:
                    debug('Found a horizontal asymptote at y=' +
                          str(ll) + ' as x->-oo')
                    poi.append(POI(0, ll, 7))
            q.put(poi)
        except NotImplementedError:
            debug('NotImplementedError for finding limit of "' +
                  str(expr) + '"')
        if poi == []:
            debug('Done calculating horizontal asymptotes.' +
                  'None found.')
        else:
            debug('Done calculating horizontal asymptotes')

    def calc_poi(self):
        expr = self.simp_expr
        self.poi = []
        # for trig functions, test common periods first
        if self.trigonometric and not self.polynomial:
            self._test_common_periods()
        #
        # y intercept
        #
        q_y = fQueue()
        p_y = fProcess(target=self._calc_y_intercept,
                       args=(q_y, expr,))
        p_y.start()
        if not self.constant:
            # calculate 1st and 2nd derivatives
            f1 = diff(expr, 'x')
            f2 = diff(f1, 'x')
            #
            # x intercepts
            #
            q_x = fQueue()
            p_x = fProcess(target=self._calc_x_intercepts,
                           args=(q_x, expr,))
            #
            # min/max
            #
            q_min_max = fQueue()
            p_min_max = fProcess(target=self._calc_min_max,
                                 args=(q_min_max, f1, expr,))
            #
            # inflection points
            #
            q_inflection = fQueue()
            p_inflection = fProcess(target=self._calc_inflection,
                                    args=(q_inflection, f2, expr,))
            #
            # vertical asymptotes
            #
            q_vertical_asym = fQueue()
            p_vertical_asym = \
                fProcess(target=self._calc_vertical_asym,
                         args=(q_vertical_asym, expr,))
            #
            # horizontal asymptotes
            #
            q_horizontal_asym = fQueue()
            p_horizontal_asym = \
                fProcess(target=self._calc_horizontal_asym,
                         args=(q_horizontal_asym, expr,))
            #
            # points where the slope is 45 degrees
            #
            q_slope_45 = fQueue()
            p_slope_45 = fProcess(target=self._calc_slope_45,
                                  args=(q_slope_45, f1, expr,))
            # start processes, different process for each POI type
            p_x.start()
            p_min_max.start()
            p_inflection.start()
            p_vertical_asym.start()
            p_horizontal_asym.start()
            p_slope_45.start()
            # get the processes output
            poi_x = q_x.get()
            poi_min_max = q_min_max.get()
            poi_inflection = q_inflection.get()
            poi_vertical_asym = q_vertical_asym.get()
            poi_horizontal_asym = q_horizontal_asym.get()
            poi_slope_45 = q_slope_45.get()
            # wait until all processes are done
            p_x.join()
            p_min_max.join()
            p_inflection.join()
            p_vertical_asym.join()
            p_horizontal_asym.join()
            p_slope_45.join()
            # gather POIs
            for i in poi_x:
                self.poi.append(i)
            for i in poi_min_max:
                self.poi.append(i)
            for i in poi_inflection:
                self.poi.append(i)
            for i in poi_vertical_asym:
                self.poi.append(i)
            for i in poi_horizontal_asym:
                self.poi.append(i)
            for i in poi_slope_45:
                self.poi.append(i)
        # Add y intercept to POIs (if any)
        poi_y = q_y.get()
        p_y.join()
        if poi_y is not None:
            self.poi.append(poi_y)
        # assign all POI to the present function
        for i in self.poi:
            i.function = self

    def check_periodic(self, x):
        l = len(x)
        if l > 1:
            for i in range(0, l - 1):
                if not self.periodic:
                    for j in range(1, l):
                        if not self.periodic:
                            for n in range(1, 11):
                                if not self.periodic:
                                    period = abs(n * (x[j] - x[i]))
                                    self._test_period(period)

    def _test_period(self, period):
        if period != 0:
            debug('Trying period: ' + str(period))
            pf = self.simp_expr.subs('x', 'x+period')
            # instead of testing if f(x)==f(x+period)
            # we're testing if f(x+period)==f(x+2*period)
            # solves problems like with sec(x) where otherwise it
            # would not find the equality
            f = pf.subs('period', period)
            g = pf.subs('period', 2 * period)
            if f - g == 0:
                debug('Function is periodic and has a period of ' +
                      str(period) + '. Smaller periods may exist.')
                self.periodic = True
                self.period = period
                margin = float(self.period * 0.6)
                self.x_min_manual = -margin
                self.x_max_manual = margin

    # checks the functions for some common periods
    # multiples of 0.25 (up to 1)
    # multiples of 1 (up to 4)
    # multiples of pi/4 (up to 2*pi)
    # multiples of pi (up to 4*pi)
    def _test_common_periods(self):
        debug('Trying some common periods to determine if ' +
              'function is periodic')
        period_list = [pi / 4, pi / 2, 2 * pi / 3, 3 * pi / 4, pi,
                       2 * pi, 3 * pi / 2, 3 * pi, 4 * pi,
                       0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
        for period in period_list:
            if not self.periodic:
                self._test_period(period)

    def _check_trigonometric(self):
        e = str(self.simp_expr)
        # only test for periods for trig functions
        if 'sin(' in e or 'cos(' in e or 'tan(' in e or \
                'cot(' in e or 'sec(' in e or 'csc(' in e:
            self.trigonometric = True
            debug('Function could be periodic.')
        else:
            self.trigonometric = False
            debug('Function cannot be periodic.')

    def __init__(self, expr, xylimits, calc_pois=True,  logscale=False):
        # the number of points to calculate within the graph using
        # the function
        self.resolution = 1000
        self.visible = True
        self.constant = False
        self.valid = True
        self.polynomial = False
        self.trigonometric = False
        self.periodic = False
        self.period = None
        self.expr = self._get_expr(expr)
        # the min and max values to use for manual POI searching
        self.x_min_manual = -20
        self.x_max_manual = 20
        self.calc_pois = calc_pois

        # simplifying helps with functions like y=x^2/x which is
        # actually just y=x. Doesn't hurt in any case.
        # Also throws an error in case there are syntax problems

        try:
            self.simp_expr = self._simplify_expr(self.expr)
            # expression as used by numpy
            self.np_expr = self._get_np_expr(str(self.simp_expr))
            self.valid = self.update_function_points(xylimits)
        except Exception as e:
            print(f'{e}')
            self.valid = False
        self.poi = []

        if self.valid:
            self.mathtex_expr = \
                self._get_mathtex_expr(self.simp_expr)
            self.polynomial = self.simp_expr.is_polynomial()
            if not self.polynomial:
                self._check_trigonometric()
            # self.calc_poi()
            if calc_pois:
                self.calc_poi()
