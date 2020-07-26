import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import log

SEED = 42

logger = log.setup_custom_logger(__name__)


class DataTransformer:

    def __init__(self):
        self.fare_median = None
        self.age_median = None
        self.embarked_most_frequent = None
        self.fare_intervals = None
        self.age_intervals = None
        self.encoders = None

    def fit(self, dataset):
        logger.info('Fitting dataset')
        self.fare_median = dataset['Fare'].median()
        self.age_median = dataset['Age'].median()
        self.embarked_most_frequent = dataset['Embarked'].mode()[0]

        self.apply_imputations(dataset)

        self.fare_intervals = pd.qcut(dataset['Fare'], 4, precision=4).unique()

        self.age_intervals = pd.cut(dataset['Age'], 5).unique()

        self.calculate_new_features(dataset)

        self.encoders = self.get_encoders(dataset)

        return self

    def apply_imputations(self, dataset):
        logger.info('Applying imputations')
        feature_imputations = self.get_feature_imputations()
        for feature in feature_imputations:
            feature_imputations[feature](dataset)

    def calculate_new_features(self, dataset):
        logger.info('Calculating new features')
        new_features = self.get_new_features()
        for feature in new_features:
            dataset[feature] = new_features[feature](dataset)

    @staticmethod
    def get_encoders(dataset):

        dataset.to_pickle('dataset.pkl')

        categorical_variables = ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']

        encoders = {}
        for cat in categorical_variables:
            encoder = LabelEncoder().fit(dataset[cat])
            encoders[cat] = encoder

        return encoders

    def get_feature_imputations(self):
        age_inputation_fn = lambda dataset: dataset['Age'].fillna(self.age_median, inplace=True)

        embarked_inputation_fn = lambda dataset: dataset['Embarked'].fillna(self.embarked_most_frequent, inplace=True)

        fare_inputation_fn = lambda dataset: dataset['Fare'].fillna(self.fare_median, inplace=True)

        feature_inputation = {'Age': age_inputation_fn, 'Embarked': embarked_inputation_fn, 'Fare': fare_inputation_fn}

        return feature_inputation

    @staticmethod
    def get_interval(value, intervals):
        for interval in intervals:
            if value in interval:
                return interval
        else:
            print(value)

    def get_new_features(self):

        family_size_feature_fn = lambda dataset: dataset['SibSp'] + dataset['Parch'] + 1

        is_alone_feature_fn = lambda dataset: (dataset.FamilySize == 1).astype(int)

        is_married_fn = lambda dataset: (dataset.Title == 'Mrs').astype(int)

        title_feature_fn = lambda dataset: dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

        fare_bin_feature_fn = lambda dataset: dataset.Fare.apply(
            lambda fare: self.get_interval(fare, self.fare_intervals))

        age_bin_feature_fn = lambda dataset: dataset.Age.apply(lambda age: self.get_interval(age, self.age_intervals))

        new_features = {'FamilySize': family_size_feature_fn, 'IsAlone': is_alone_feature_fn, 'Title': title_feature_fn,
                        'FareBin': fare_bin_feature_fn, 'AgeBin': age_bin_feature_fn, 'IsMarried': is_married_fn}

        return new_features

    def transform(self, df):
        logger.info('Transforming dataset')
        dataset = df.copy(deep=True)

        self.apply_imputations(dataset)

        self.calculate_new_features(dataset)

        dataset['Sex_Code'] = self.encoders['Sex'].transform(dataset['Sex'])
        dataset['Embarked_Code'] = self.encoders['Embarked'].transform(dataset['Embarked'])
        dataset['Title_Code'] = self.encoders['Title'].transform(dataset['Title'])
        dataset['AgeBin_Code'] = self.encoders['AgeBin'].transform(dataset['AgeBin'])
        dataset['FareBin_Code'] = self.encoders['FareBin'].transform(dataset['FareBin'])

        features = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'AgeBin_Code', 'FareBin_Code', 'FamilySize',
                    'IsAlone', 'IsMarried']

        return pd.get_dummies(dataset[features])

    def save(self, filename):
        logger.info('Saving transformer in file: {}'.format(filename))
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        logger.info('Loading transformer from file: {}'.format(filename))
        with open(filename, "rb") as f:
            return pickle.load(f)
