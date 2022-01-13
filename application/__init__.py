from flask import Flask
from flask_cors import CORS
from application.blueprints.graph import graph


def create_app():
    """
    Create a Flask application using the application factory pattern.

    :return: Flask application
    """
    app = Flask(__name__, instance_relative_config=False)
    CORS(app)  # this makes the CORS feature cover all routes in the app

    # configuring the app
    app.config.from_object('config.DevConfig')

    # instance configurations
    app.config.from_pyfile('config.py', silent=True)

    # register blueprints
    app.register_blueprint(graph)

    return app
