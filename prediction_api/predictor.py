import log
import joblib
from trainer.transformer import DataTransformer
import pandas as pd


logger = log.setup_custom_logger(__name__)


class Predictor:
    def __init__(self, name, classifier, transformer):
        self.name = name
        self.classifier = classifier
        self.transformer = transformer

    def passenger_to_pandas(self, passenger):
        data = pd.DataFrame(columns=list(passenger.keys()), data=[list(passenger.values())])
        features = self.transformer.transform(data)
        return features

    def predict(self, passenger):
        features = self.passenger_to_pandas(passenger)
        logger.info(features)
        prediction = self.classifier.predict(features)
        return prediction


actual_predictors = []


def build_predictor(name):
    classifier = joblib.load('trainer/models/{}'.format(name))
    transformer = DataTransformer.load('trainer/models/transformer')
    actual_predictors.append(Predictor(name, classifier, transformer))