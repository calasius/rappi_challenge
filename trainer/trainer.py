from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from transformer import DataTransformer
import pandas as pd
import numpy as np
import joblib
import log
from system_usage_profiling import SystemUsageProfiler

logger = log.setup_custom_logger(__name__)

SEED = 42


def train_model():
    train = pd.read_csv('data/train.csv', index_col=0)
    test = pd.read_csv('data/test.csv', index_col=0)

    transformer = DataTransformer().fit(pd.concat([train, test]))

    X_train = transformer.transform(train)
    y_train = train.Survived

    classifier = RandomForestClassifier(criterion='gini',
                                        n_estimators=1750,
                                        max_depth=7,
                                        min_samples_split=6,
                                        min_samples_leaf=6,
                                        max_features='auto',
                                        oob_score=True,
                                        random_state=SEED,
                                        n_jobs=-1,
                                        verbose=0)

    N = 5
    oob = 0

    scores, acc_scores = [], []

    skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        # Fitting the model
        classifier.fit(X_train.iloc[trn_idx], y_train.iloc[trn_idx])

        # Computing Validation AUC score
        val_fpr, val_tpr, val_thresholds = roc_curve(y_train.iloc[val_idx],
                                                     classifier.predict_proba(X_train.iloc[val_idx])[:, 1])
        val_auc_score = auc(val_fpr, val_tpr)

        scores.append(val_auc_score)

        acc_scores.append(accuracy_score(y_train.iloc[val_idx], classifier.predict(X_train.iloc[val_idx])))

        oob += classifier.oob_score_ / N
        logger.info('Fold {} OOB Score: {}'.format(fold, classifier.oob_score_))

    logger.info('Average OOB Score: {}'.format(oob))
    logger.info('Average auc: {}'.format(np.mean(val_auc_score)))
    logger.info('Average accuracy: {}'.format(np.mean(acc_scores)))

    return classifier, transformer


if __name__ == '__main__':
    profiler = SystemUsageProfiler("train_process", update_period=1)
    filename = 'models/random_forest_classifier'
    model, transformer = train_model()
    logger.info('Saving model in file: {}'.format(filename))
    joblib.dump(model, filename)
    transformer.save('models/transformer')
    profiler.finish()

