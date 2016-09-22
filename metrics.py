from sklearn.utils import check_array
import numpy as np


def geh(y_true, y_pred):
    return np.sqrt(2*np.power(y_pred - y_true, 2)/(y_pred + y_true)).mean(axis=0)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Use of this metric is not recommended; for illustration only.
    See other regression metrics on sklearn docs:
      http://scikit-learn.org/stable/modules/classes.html#regression-metrics
    Use like any other metric
    >>> y_true = [3, -0.5, 2, 7]; y_pred = [2.5, -0.3, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    Out[]: 24.791666666666668
    """

    y_true, y_pred = check_array(y_true), check_array(y_pred)

    # Note: does not handle mix 1d representation
    # if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean(axis=0))[0]


def MASE(training_series, testing_series, prediction_series):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forecast error for univariate time series prediction.

    See "Another look at measures of forecast accuracy", Rob J Hyndman

    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.

    """
    # print("Needs to be tested.")
    n = training_series.shape[0]
    d = np.abs(np.diff(training_series)).sum() / (n - 1)

    errors = np.abs(testing_series - prediction_series)
    return errors.mean() / d


def median_percentage_error(y_pred, y_test):
    return np.median(np.abs((y_test - y_pred) / y_test)) * 100


def smape(y_pred, y_test):
    return np.mean(np.abs(y_pred - y_test) / ((np.abs(y_pred) + np.abs(y_pred)) / 2))
