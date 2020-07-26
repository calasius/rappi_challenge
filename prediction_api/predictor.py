import log
import joblib
from trainer.transformer import DataTransformer


logger = log.setup_custom_logger(__name__)


class Predictor:
    def __init__(self, name, classifier, transformer):
        self.name = name
        self.classifier = classifier
        self.transformer = transformer

    @staticmethod
    def passenger_to_pandas(passenger):
        return None

    def predict(self, passenger):
        features = self.passenger_to_pandas(passenger)
        prediction = self.classifier.predict(features)
        return prediction


actual_predictors = []


def build_predictor(name):
    classifier = joblib.load('trainer/models/{}'.format(name))
    transformer = DataTransformer.load('trainer/models/transformer')
    actual_predictors.append(Predictor(name, classifier, transformer))