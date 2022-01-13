#!/usr/bin/env python
# vim:et:sta:sts=4:sw=4:ts=8:tw=79:

# pod function borrowed from the sympy project bug tracker.
# It will be available in a future sympy release. See:
# http://goo.gl/7HdtGJ
#
# sample function adapted from: http://goo.gl/Ue1r6K (CC0 licensed)
#

from __future__ import division
from sympy import Wild, solve, simplify, log, exp, re, im
from logging import debug
import numpy as np
from queue import Empty as qEmpty

win32 = True
try:
    import winshell
except ImportError:
    win32 = False

# don't use multiprocessing on win32. It causes huge slowdowns
# because of the lack of os.fork()
if win32:
    from threading import Thread as fProcess
    from queue import Queue as fQueue
else:
    from multiprocessing import Process as fProcess
    from multiprocessing import Queue as fQueue


class BreakLoop(Exception):
    """
    An Exception class to help breaking out of nested loops
    """
    pass


def pod(expr, sym):
    """
    Find the points of Discontinuity of a real univariate function

    Example
    =========

    # >>> from sympy.calculus.discontinuity import pod
    # >>> from sympy import exp, log, Symbol
    # >>> x = Symbol('x', real=True)
    # >>> pod(log((x-2)**2) + x**2, x)
    # [2]
    # >>> pod(exp(1/(x-2)**2), x)
    [2]

    """
    #  For now this a hacky implementation.

    # not trying for trig function because they have infinitely
    # many solutions, and this can turn out to be problematic
    # for example solve(tan(x), x) returns only 0 for now
    if expr.is_polynomial():
        return []
    pods = []
    try:
        pods = pods + solve(simplify(1 / expr), sym)
    except NotImplementedError:
        return []
    p = Wild("p")
    q = Wild("q")
    r = Wild("r")

    # check the condition for log
    expr_dict = expr.match(r * log(p) + q)
    if not expr_dict[r].is_zero:
        pods += solve(expr_dict[p], sym)
        pods += pod(expr_dict[p], sym)
        pods += pod(expr_dict[r], sym)

    # check the condition for exp
    expr = expr.rewrite(exp)
    expr_dict = expr.match(r * exp(p) + q)
    if not expr_dict[r].is_zero:
        pods += solve(simplify(1 / expr_dict[p]), sym)
        pods += pod(expr_dict[p], sym)
        pods += pod(expr_dict[r], sym)

    return list(set(pods))  # remove duplicates


def mpsolve(q, expr):
    try:
        x = solve(expr, 'x')
        q.put(x)
    except NotImplementedError:
        debug('NotImplementedError for solving "' + str(expr) + '"')
        q.put(None)
    except TypeError:
        debug('TypeError exception. This was not supposed to ' +
              'happen. Probably a bug in sympy.')
        q.put(None)


def fsolve(expr):
    xl = []
    try:
        q = fQueue()
        p = fProcess(target=mpsolve, args=(q, expr,))
        p.start()
        # timeout solving after 5 seconds
        x = q.get(True, 5)
        p.join()
        if x is None:
            xl = None
        else:
            for i in x:
                xc = rfc(i)
                if xc is not None:
                    xl.append(xc)
                    debug('Found solution: ' + str(xc))
    except qEmpty:
        debug('Solving timed out.')
        xl = None
    finally:
        # if we're on windows and using threading instead
        # of multiprocessing, terminate() doesn't work
        if not win32:
            p.terminate()
    if xl == []:
        xl = None
    return xl


def rfc(x):
    """ rfc - Real From Complex
    This tries to detect if a complex number as given by sympy is
    actually a real number. If it is, then it returns the real part
    as a float.
    """
    try:
        xc = round(float(x), 15)
    # TypeError is thrown for complex solutions during casting
    # to float. We only want real solutions.
    except TypeError:
        # But, there are times that a real solution is calculated
        # as a complex solution, with a really small imaginary part,
        # for example: 2.45009658902771 - 1.32348898008484e-23*I
        # so in cases where the imaginary part is really small,
        # keep only the real part as a solution
        xe = x.evalf()
        debug('Checking if this is a complex number: ' + str(xe))
        real = re(xe)
        img = im(xe)
        if abs(img) < 0.00000000000000001 * abs(real):
            debug(str(real) + ' is actually a real.')
            xc = round(float(real), 15)
        else:
            debug('Yes, it is probably a complex.')
            xc = None
    return xc


def percentile(plist, perc):
    """
    returns the perc (range 0-100) percentile of plist
    """
    x = sorted(plist)
    n = len(x)
    pos = (n + 1) * perc / 100
    k = int(np.floor(pos))
    a = pos - k
    p = x[k - 1] + a * (x[k] - x[k - 1])
    return p


def remove_outliers(plist):
    """
    This function takes a list (of floats/ints) and returns the list,
    having replaced any outliers with the median value of the list.
    """
    q1 = percentile(plist, 25)
    q3 = percentile(plist, 75)
    iqr = q3 - q1
    # This looks like a nice value. Decrease to make it easier for
    # a value to become an outlier, increase to make it harder.
    k = 18
    # this is the median
    m = percentile(plist, 50)
    # these are the limits we don't allow values to go under/over
    min_lim = q1 - k * iqr
    max_lim = q3 + k * iqr
    if min_lim < max_lim:
        debug('Any values<' + str(min_lim) + ' or >' +
              str(max_lim) + ' are outliers.')
        for i in range(0, len(plist)):
            if plist[i] < min_lim or plist[i] > max_lim:
                debug('Found outlier: ' + str(plist[i]))
                # if outliers are detected, replace their values with
                # the median. That way it's easier to just set the
                # axis limits to the min/max of the remaining values.
                plist[i] = m
    return plist


def log10(f):
    return log(f, 10)


def sample(npfunc, points, tol=0.001, min_points=16, max_level=32,
           sample_transform=None):
    """
    Sample a 1D function to given tolerance by adaptive subdivision.

    The result of sampling is a set of points that, if plotted,
    produces a smooth curve with also sharp features of the function
    resolved.

    Parameters
    ----------
    npfunc : a string representing a numpy function.
        Example 'np.sin(x)'
    points : array-like, 1D
        Initial points to sample, sorted in ascending order.
        These will determine also the bounds of sampling.
    tol : float, optional
        Tolerance to sample to. The condition is roughly that the
        total length of the curve on the (x, y) plane is computed
        up to this tolerance.
    min_points : int, optional
        Minimum number of points to sample.
    max_level : int, optional
        Maximum subdivision depth.
    sample_transform : callable, optional
        Function w = g(x, y). The x-samples are generated so that w
        is sampled.

    Returns
    -------
    x : ndarray
        X-coordinates
    y : ndarray
        Corresponding values of func(x)

    Notes
    -----
    This routine is useful in computing functions that are expensive
    to compute, and have sharp features --- it makes more sense to
    adaptively dedicate more sampling points for the sharp features
    than the smooth parts.

    Examples
    --------
    # >>> def func(x):
    # ...     '''Function with a sharp peak on a smooth background'''
    # ...     a = 0.001
    # ...     return x + a**2/(a**2 + x**2)
    # ...
    # >>> x, y = sample_function(func, [-1, 1], tol=1e-3)
    #
    # >>> import matplotlib.pyplot as plt
    # >>> xx = np.linspace(-1, 1, 12000)
    # >>> plt.plot(xx, func(xx), '-', x, y[0], '.')
    # >>> plt.show()

    """
    func = _func(npfunc)
    return _sample_function(func, points, values=None, mask=None, depth=0, tol=tol, min_points=min_points,
                            max_level=max_level, sample_transform=sample_transform)


def _sample_function(func, points, values=None, mask=None, tol=0.05,
                     depth=0, min_points=16, max_level=16,
                     sample_transform=None):
    points = np.asarray(points)

    if values is None:
        values = np.atleast_2d(func(points))

    if depth > max_level:
        # recursion limit
        return points, values

    if mask is None:
        x_a = points[..., :-1][...]
        x_b = points[..., 1:][...]
    else:
        x_a = points[..., :-1][..., mask]
        x_b = points[..., 1:][..., mask]

    x_c = .5 * (x_a + x_b)
    y_c = np.atleast_2d(func(x_c))

    x_2 = np.r_[points, x_c]
    y_2 = np.r_['-1', values, y_c]
    j = np.argsort(x_2)

    x_2 = x_2[..., j]
    y_2 = y_2[..., j]

    # -- Determine the intervals at which refinement is necessary

    if len(x_2) < min_points:
        mask = np.ones([len(x_2) - 1], dtype=bool)
    else:
        # represent the data as a path in N dimensions
        # (scaled to unit box)
        if sample_transform is not None:
            y_2_val = sample_transform(x_2, y_2)
        else:
            y_2_val = y_2

        p = np.r_['0',
                  x_2[None, :],
                  y_2_val.real.reshape(-1, y_2_val.shape[-1]),
                  y_2_val.imag.reshape(-1, y_2_val.shape[-1])
                  ]

        sz = (p.shape[0] - 1) // 2

        xscale = x_2.ptp(axis=-1)
        yscale = abs(y_2_val.ptp(axis=-1)).ravel()

        p[0] /= xscale
        p[1:sz + 1] /= yscale[:, None]
        p[sz + 1:] /= yscale[:, None]

        # compute the length of each line segment in the path
        dp = np.diff(p, axis=-1)
        s = np.sqrt((dp**2).sum(axis=0))
        s_tot = s.sum()

        # compute the angle between consecutive line segments
        dp /= s
        dcos = np.arccos(np.clip((dp[:, 1:] * dp[:, :-1]).sum(axis=0),
                                 -1, 1))

        # determine where to subdivide: the condition is roughly that
        # the total length of the path (in the scaled data) is
        # computed to accuracy `tol`
        dp_piece = dcos * .5 * (s[1:] + s[:-1])
        mask = (dp_piece > tol * s_tot)

        mask = np.r_[mask, False]
        mask[1:] |= mask[:-1].copy()

    # -- Refine, if necessary

    if mask.any():
        return _sample_function(func, x_2, y_2, mask, tol=tol,
                                depth=depth + 1, min_points=min_points,
                                max_level=max_level,
                                sample_transform=sample_transform)
    else:
        return x_2, y_2


def _func(npexpr):
    """
    Returns a numpy expression as a callable function
    """
    def f(x):
        return eval(npexpr)
    return f


def keep10(lst):
    """
    For a list that has more than 10 elements, keep only
    10, with uniform distribution.
    """
    length_poi = len(lst)
    if length_poi > 10:
        debug('Too many POI in list (' + str(length_poi) + '). Keeping only 10.')
        lst = [lst[0], lst[int(length_poi / 10)], lst[int(length_poi / 5)],
               lst[int(3 * length_poi / 10)], lst[int(2 * length_poi / 5)],
               lst[int(length_poi / 2)], lst[int(3 * length_poi / 5)],
               lst[int(7 * length_poi / 10)], lst[int(4 * length_poi / 5)],
               lst[-1]]
    return lst
