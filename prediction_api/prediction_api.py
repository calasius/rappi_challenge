from flask import Flask, request, Response
import log
from metrics import system_resource_metrics
from predictor import build_predictor
from errors import ServiceException
from flask import jsonify
from model import PassengerSchema
from predictor import actual_predictors

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
        raise ServiceException(message)


@app.route('/model/predict', methods=['POST'])
def predict():
    if not request.json:
        raise ServiceException('Please post a correct json')
    else:
        schema = PassengerSchema()
        errors = schema.validate(request.json)
        if len(errors) > 0:
            logger.error(errors)
            raise ServiceException(str(errors))
        else:
            passenger = schema.load(request.json)
            predictor = actual_predictors[0]
            prediction = predictor.predict(passenger.to_dict())[0]
            return Response(str(prediction), 200)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
