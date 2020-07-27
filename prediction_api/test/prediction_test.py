from predictor import build_predictor
import unittest
import json
from model import PassengerSchema

predictor = build_predictor('random_forest_classifier')
schema = PassengerSchema()


def get_passenger_from_file(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return schema.load(data)


class PredictionTest(unittest.TestCase):
    def test_survive_1(self):
        """
        Testing model prediction with a passenger that survive
        """
        passenger = get_passenger_from_file('test/passenger1.json')
        prediction = predictor.predict(passenger.to_dict())
        self.assertEqual(1, prediction)

    def test_survive_2(self):
        """
        Testing model prediction with a passenger that survive
        """
        passenger = get_passenger_from_file('test/passenger2.json')
        prediction = predictor.predict(passenger.to_dict())
        self.assertEqual(1, prediction)

    def test_survive_3(self):
        """
        Testing model prediction with a passenger that not survive
        """
        passenger = get_passenger_from_file('test/passenger3.json')
        prediction = predictor.predict(passenger.to_dict())
        self.assertEqual(0, prediction)
