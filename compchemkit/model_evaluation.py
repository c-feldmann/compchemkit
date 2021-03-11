from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from classifier import TanimotoKNN
from sklearn.model_selection import GridSearchCV
import numpy as np
from typing import *
import pandas as pd

from .data_storage import DataSet

# TODO: Maybe an own metric class which can be fed to visualize_metrics?


def evaluate_model(model, training_set: DataSet, test_set: DataSet) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not hasattr(model, 'predict'):
        TypeError("Model has no predict function")

    y_pred_training = model.predict(training_set.feature_matrix)
    y_pred_test = model.predict(test_set.feature_matrix)

    is_knn = False
    if isinstance(model, KNeighborsClassifier):
        is_knn = True
    if isinstance(model, GridSearchCV):
        if isinstance(model.estimator, KNeighborsClassifier):
            is_knn = True

    if not is_knn:
        y_score_training = model.predict_proba(training_set.feature_matrix)[:, 1]
        y_score_test = model.predict_proba(test_set.feature_matrix)[:, 1]
    else:
        y_score_training = None
        y_score_test = None

    training_metrics = evaluate_prediction(training_set.label, y_pred_training, y_score_training)
    training_metrics["data_set"] = "training"

    test_metrics = evaluate_prediction(test_set.label, y_pred_test, y_score_test)
    test_metrics["data_set"] = "training"

    return training_metrics, test_metrics


def evaluate_dataset(model, dataset: DataSet) -> pd.DataFrame:
    if not hasattr(model, 'predict'):
        TypeError("Model has no predict function")

    y_pred = model.predict(dataset.feature_matrix)

    is_knn = False
    if isinstance(model, KNeighborsClassifier) or isinstance(model, TanimotoKNN):
        is_knn = True
    if isinstance(model, GridSearchCV):
        if isinstance(model.estimator, KNeighborsClassifier) or isinstance(model.estimator, TanimotoKNN):
            is_knn = True

    if not is_knn:
        y_score = model.predict_proba(dataset.feature_matrix)[:, 1]
    else:
        y_score = None

    metric_list = evaluate_prediction(dataset.label, y_pred, y_score)

    return metric_list


def evaluate_prediction(y_true, y_predicted, y_score=None, nantozero=False) -> pd.DataFrame:
    if len(y_true) != len(y_predicted):
        raise IndexError("y_true and y_predicted are not of equal size!")
    if y_score is not None:
        if len(y_true) != len(y_score):
            raise IndexError("y_true and y_score are not of equal size!")

    fill = 0 if nantozero else np.nan

    if sum(y_predicted) == 0:
        mcc = fill
        precision = fill
    else:
        mcc = metrics.matthews_corrcoef(y_true, y_predicted)
        precision = metrics.precision_score(y_true, y_predicted)

    result_list = [{"metric": "MCC", "value": mcc},
                   {"metric": "F1", "value": metrics.f1_score(y_true, y_predicted)},
                   {"metric": "BA", "value": metrics.balanced_accuracy_score(y_true, y_predicted)},
                   {"metric": "Precision", "value": precision},
                   {"metric": "Recall", "value": metrics.recall_score(y_true, y_predicted)},
                   {"metric": "set_size", "value": y_true.shape[0]},
                   {"metric": "pos_true", "value": len([x for x in y_true if x == 1])},
                   {"metric": "neg_true", "value": len([x for x in y_true if x == 0])},
                   {"metric": "pos_predicted", "value": len([x for x in y_predicted if x == 1])},
                   {"metric": "neg_predicted", "value": len([x for x in y_predicted if x == 0])},
                   ]

    if y_score is not None:
        result_list.append({"metric": "AUC", "value": metrics.roc_auc_score(y_true, y_score)})
    else:
        result_list.append({"metric": "AUC", "value": np.nan})

    return pd.DataFrame(result_list)
