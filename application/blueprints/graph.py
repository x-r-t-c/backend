import numpy as np
from numpy import inf
from sys import float_info
from flask import Blueprint, request, make_response, jsonify
from application.functionplot.FunctionGraph import FunctionGraph

graph = Blueprint('graph', __name__)


@graph.route('/graph/update-points', methods=['POST'])
def handle_update():
    headers = {"Content-Type": "application/json",
               "method": "POST",
               "mode": "cors",
               "Access-Control-Allow-Origin": "http://127.0.0.1:8080",
               "Access-Control-Allow-Methods": "POST"
               }
    if request.method == 'POST':
        data = request.get_json()
        limits = data['limits']
        expr = data['expr']
        # print(limits, expr)
        f_graph = FunctionGraph()
        f_graph.limits = limits
        f_graph.add_function(expr, True)
        f = next(filter(lambda g: g.expr == expr, f_graph.functions), None)
        if f is not None:
            f.update_function_points(limits)
            # f_graph.update_graph_points()
            x, y = f.graph_points
            x = list(x)
            y = list(y)
            for i in range(0, len(y)):
                if y[i] == np.inf:
                    y[i] = float_info.max
                elif y[i] == -np.inf:
                    y[i] == float_info.min
            rsp = [f_graph.get_limits(), x, y]
        return make_response(jsonify(rsp), 200, headers)


@graph.route('/graph', methods=['POST'])
def handle_request():
    headers = {"Content-Type": "application/json",
               "method": "POST",
               "mode": "cors",
               "Access-Control-Allow-Origin": "http://127.0.0.1:8080",
               "Access-Control-Allow-Methods": "POST"
               }

    if request.method == 'POST':
        data = request.get_json()

        f_graph = FunctionGraph()

        # add functions to FunctionGraph
        no_of_functions = len(data['functions'])

        for i in range(0, no_of_functions):
            expr = data['functions'][i]['expr']
            has_poi = data['functions'][i]['hasPoi']

            # if poi have been calculated before do not recalculate
            if not has_poi:
                f_graph.add_function(expr, True)
                f = next(filter(lambda g: g.expr == expr, f_graph.functions), None)
                pois = []
                if f is not None:
                    for p in f.poi:
                        poi = create_json_poi(p)
                        pois.append(poi)

                data['functions'][i]['poi'] = pois
                data['functions'][i]['hasPoi'] = True
            else:
                f_graph.add_function(expr, False)

        for i in range(0, no_of_functions):
            expr = data['functions'][i]['expr']
            f = next(filter(lambda g: g.expr == expr, f_graph.functions), None)
            x, y = f.graph_points
            data['functions'][i]['graphPoints'] = get_graph_points(x, y)

        graph_pois = []
        for p in f_graph.poi:
            poi = create_json_poi(p)
            graph_pois.append(poi)

        data['graphPoi'] = graph_pois

        data['limits'] = f_graph.get_limits()

        return make_response(data, 200, headers)


def create_json_function(function, visible=True, has_pois=True):
    # expression
    expr = function.expr

    # add pois
    pois = []
    for point in function.poi:
        json_poi = {
            'x': point.x,
            'y': point.y,
            'size': 1,
            'function': function.expr,
            'color': None,
            'point_type': point.point_type
        }
        pois.append(json_poi)

    # add graph points
    x, y = function.graph_points
    x = list(x)
    y = list(y)

    # handle inf and -inf that cause problem in JSON.parse
    x_temp = []
    y_temp = []
    for i in range(len(x)):
        if y[i] == inf:
            y[i] = float_info.max
        elif y[i] == -inf:
            y[i] = float_info.min
        x_temp.append(x[i])
        y_temp.append(y[i])
    graph_points = [x_temp, y_temp]

    # add visibility
    visibility = visible
    resolution = function.resolution
    json_function = {
        "expr": expr,
        "poi": pois,
        "graph_points": graph_points,
        "visible": visibility,
        "color": None,
        "has_pois": has_pois,
        "resolution": resolution
    }
    print(resolution)
    return json_function


def create_json_poi(p):
    x = p.x
    y = p.y
    point_type = p.point_type
    size = p.size
    color = p.color
    poi = {
        "x": x,
        "y": y,
        "point_type": point_type,
        "size": size,
        "color": color
    }
    return poi


def get_graph_points(x, y):
    # add graph points
    x = list(x)
    y = list(y)

    # handle inf and -inf that cause problem in JSON.parse
    x_temp = []
    y_temp = []
    for i in range(len(x)):
        if y[i] == inf:
            y[i] = float_info.max
        elif y[i] == -inf:
            y[i] = float_info.min
        x_temp.append(x[i])
        y_temp.append(y[i])
    graph_points = [x_temp, y_temp]
    return graph_points
