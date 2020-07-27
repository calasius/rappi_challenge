from marshmallow import Schema, fields, post_load

all_fields = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']


class Passenger(object):
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def to_dict(self):
        self_fields = vars(self)
        for field in all_fields:
            if field not in self_fields:
                self_fields[field] = None
        return self_fields


class PassengerSchema(Schema):
    Pclass = fields.Integer(required=True)
    Name = fields.String(required=False)
    Sex = fields.String(required=True)
    Age = fields.Integer(required=False)
    SibSp = fields.Integer(required=True)
    Parch = fields.Integer(required=True)
    Ticket = fields.String(required=False)
    Fare = fields.Float(required=False)
    Cabin = fields.String(required=False)
    Embarked = fields.String(required=False)

    @post_load
    def post_load(self, data, **kwargs):
        return Passenger(**data)
