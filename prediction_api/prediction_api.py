from flask import Flask, render_template, request, redirect, url_for, Response, abort
import log
from metrics import system_resource_metrics
from predictor import build_predictor
from errors import ServiceException
from flask import jsonify
from model import PassengerSchema, Passenger

CONTENT_TYPE_LATEST = str('text/plain; version=0.0.4; charset=utf-8')

logger = log.setup_custom_logger(__name__)

app = Flask(__name__)


@app.errorhandler(ServiceException)
def handle_invalid_usage(error):
    logger.error(error.message)
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/metrics', methods=['GET'])
def get_data():
    return Response(system_resource_metrics.collect_metrics(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/model/deploy/<name>', methods=['GET'])
def deploy(name):
    try:
        build_predictor(name)
        return Response('The model {} has been deployed correctly'.format(name), status=200)
    except Exception as e:
        message = 'There was a problem deploying model {} error = {}'.format(name, str(e))
        raise ServiceException(message, status_code=410)


@app.route('/model/predict/', methods=['POST'])
def predict():
    if not request.json:
        abort(400)
    else:
        schema = PassengerSchema()
        errors = schema.validate(request.json)
        logger.info(errors)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
