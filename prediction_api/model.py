from marshmallow import Schema, fields, post_load


class Passenger(object):
    def __init__(self, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked):
        self.pclass = pclass
        self.name = name
        self.sex = sex
        self.age = age
        self.sibsp = sibsp
        self.parch = parch
        self.ticket = ticket
        self.fare = fare
        self.cabin = cabin
        self.embarked = embarked


class PassengerSchema(Schema):
    pclass = fields.String(required=True)
    name = fields.String(required=False)
    sex = fields.String(required=True)
    age = fields.Integer(required=False)
    sibsp = fields.Integer(required=True)
    parch = fields.Integer(required=True)
    ticket = fields.String(required=False)
    fare = fields.Float(required=False)
    embarked = fields.Integer(required=False)

    @post_load
    def post_load(self, data, **kwargs):
        return Passenger(**data)
