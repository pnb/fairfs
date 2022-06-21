# Object for selecting columns from dataset using threshold in scikit-learn pipelines.

import math
from sklearn.base import BaseEstimator


class ColumnThresholdSelector(BaseEstimator):
    """Object for selecting columns from a dataset given a threshold.

    Parameters:
        columnwise_values (DataFrame): DataFrame containing fairness calculation for each column

    Returns:
        Index: Pandas Index of columns to use

    """


def __init__(self, col_values=None, cutoff_value=None, metric=None):
    self.col_values = col_values
    self.cutoff_value = cutoff_value
    self.metric = metric


def select_features(self):
    """Take fairness values for each column and use the given metric to remove.

    most unfair features using cutoff. "Most unfair" is measured by absolute distance
    from 1 (which represents the classes being exactly equal)

    Args:
        columnwise_values (DataFrame): DataFrame containing fairness calculation for each column

    Returns:
        Index: Pandas Index of columns to use

    """
    # calculate relative unfairness by getting distance of each value from 1
    adjusted_values = abs(self.col_values - 1)

    # Get columns sorted for relevant metric (transposed to allow dropping)
    sorted_cols = adjusted_values.T.sort_values(by=[self.metric])

    # Get number of columns to drop
    cutoff_index = math.floor(self.cutoff_value * len(sorted_cols))

    # select only desired columns by cutting off highest value rows
    new_cols = sorted_cols.iloc[:-cutoff_index]

    # return list of columns as Index object
    return new_cols.T.columns


def fit(self):
    pass


def transform(self):
    pass


def fit_transform(self):
    pass
