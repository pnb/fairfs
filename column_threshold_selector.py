# Object for selecting columns from dataset using threshold in scikit-learn pipelines.
import pandas as pd
import numpy as np
import math
import shap
from sklearn import metrics, model_selection, pipeline, preprocessing
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnThresholdSelector(BaseEstimator, TransformerMixin):
    """Object for selecting columns from a dataset given a threshold.

    Parameters:
        columnwise_values (DataFrame): DataFrame containing fairness calculation for each column

    Returns:
        Index: Pandas Index of columns to use

    """

    def __init__(self, model, privileged_value, cutoff_value, unfairness_metric):
        self.model = model
        # self.sensitive_column = sensitive_column
        self.privileged_value = privileged_value
        self.cutoff_value = cutoff_value
        # string or function. if string match to existing function or people can pass in their own function
        self.unfairness_metric = unfairness_metric
        self.selected_features = []
        # self.accuracy = []
        self.shap_values = pd.DataFrame()
        self.pred_labels = pd.DataFrame()

    def fit(self, X, y=None):
        """ actual fitting of model,
        choosing features given parameters, etc
        mostly will be moved over from run_model in fair_shap.py
        """
        assert isinstance(X, pd.DataFrame), 'Only pd.DataFrame inputs for X are supported'
        # Create the DataFrame to hold the SHAP results, labels, and accuracy
        self.shap_values = pd.DataFrame(index=X.index, columns=X.columns)
        self.pred_labels = pd.DataFrame(index=X.index, columns=['labels'])

        # Create cross-validation train test split
        cross_val = model_selection.KFold(4, shuffle=True, random_state=11798)

        for train_index, test_index in cross_val.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_test = pd.DataFrame(X_test, columns=X.columns)

            # Run the model as defined in the constants, get predictions and accuracy
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)

            # self.accuracy.append(pd.Series({
            #     'label_type': 'y',
            #     'auc': ACCURACY_METRIC(y_test, predictions)
            # }))

            # Get shap values
            explainer = shap.TreeExplainer(self.model)
            current_shap_values = explainer.shap_values(X_test)
            # Only need to save one end of the range of values
            self.shap_values.iloc[test_index] = current_shap_values[0]

            # Get predicted labels
            self.pred_labels.iloc[test_index] = predictions[0]

        # feature selection
        fairness_values = calc_feature_fairness_scores(self, X_train, y_train, self.shap_values)

        # select features
        self.selected_features = select_features(self, fairness_values, self.cutoff_value)

        return self

    def transform(self, X, y=None):
        """ return data with only fair features included. y is optional for unsupervised models
        """
        return X[self.selected_features]

    def fit_transform(self, X, y=None):
        #call fit and then call transform
        return self.fit(X, y).transform(X)


def select_features(self, columnwise_values, cutoff_value):
    """Take fairness values for each column and use the given metric to remove.

    most unfair features using cutoff. "Most unfair" is measured by absolute distance
    from 1 (which represents the classes being exactly equal)

    Args:
        columnwise_values (DataFrame): DataFrame containing fairness calculation for each column
        cutoff_value (Float): Percentage of columns to keep
    Returns:
        Index: Pandas Index of columns to use

    """
    # calculate relative unfairness by getting distance of each value from 1
    adjusted_values = abs(columnwise_values - 1)

    # Get columns sorted for relevant metric (transposed to allow dropping)
    sorted_cols = adjusted_values.T.sort_values(by=[self.unfairness_metric])

    # Get number of columns to drop
    cutoff_index = math.floor(self.cutoff_value * len(sorted_cols))

    # select only desired columns by cutting off highest value rows
    new_cols = sorted_cols.iloc[:-cutoff_index]

    # return list of columns as Index object
    return new_cols.T.columns


def convert_shap_to_bools(data, shap_values):
    """
    Convert shapley values to booleans. If shapley value is less than 0, then
    set to 0. If greater than or equal to 0, set to 1. This allows for
    traditional fairness metrics to be applied to shapley values.

    Args:
        data (DataFrame): original dataset used for train and test
        shap_values (DataFrame): shapley values for the model (same size as data)

    Returns:
        DataFrame: converted shapley values for the dataset

    """
    converted_df = pd.DataFrame(index=data.index, columns=data.columns)
    for column in shap_values:
        values = []
        for value in shap_values[column]:
            values.append(0 if value < 0 else 1)

        converted_df[column] = values

    return converted_df


def calc_feature_fairness_scores(self, data, labels, shap_values):
    """
    Calculate fairness scores for the existing metrics and store for each column.

    Args:
        data (DataFrame): original dataset used for train and test
        labels (Array): list of truth labels for the data
        shap_values (DataFrame): shapley labels for the model
    Returns:
        DataFrame: fairness scores for each column

    """
    cols = data.columns
    all_fairness_scores = pd.DataFrame(index=[self.unfairness_metric], columns=cols)
    converted_df = convert_shap_to_bools(data, shap_values)

    priv_indices = (pd.DataFrame((
        [data.loc[data[PROTECTED_COLUMN] == self.privileged_value]])[0])).index
    priv_truth = labels[priv_indices]

    unpriv_indices = (pd.DataFrame((
        [data.loc[data[PROTECTED_COLUMN] != self.privileged_value]])[0])).index
    unpriv_truth = labels[unpriv_indices]

    for col in cols:
        # Get predictions for privileged and unprivileged classes
        priv_predict = converted_df[col][priv_indices].tolist()
        unpriv_predict = converted_df[col][unpriv_indices].tolist()

        # Get statistical parity (ratio of marginal distributions)
        # close to 1 means marginal distributions are equal
        # less than 1 means priv group has fewer predicted pos than unpriv
        # greater than 1 means priv group has more predicted pos than unpriv
        if self.unfairness_metric == 'stat_parity':
            priv_marg_dist = marginal_dist(priv_predict)
            unpriv_marg_dist = marginal_dist(unpriv_predict)
            all_fairness_scores[col]['stat_parity'] = priv_marg_dist / unpriv_marg_dist

        # Get treatment score, stored as ratio for comparison purposes
        # greater than 1 means unpriv group has greater false neg to false pos ratio than priv group
        # 1 is equal rates of false neg to false pos for both groups
        # less than 1 means priv group has greater false neg to false pos ratio than unpriv group
        elif self.unfairness_metric == 'treatment_eq_ratio':
            priv_treatment_eq = treatment_score(priv_truth, priv_predict)
            unpriv_treatment_eq = treatment_score(unpriv_truth, unpriv_predict)
            all_fairness_scores[col]['treatment_eq_ratio'] = priv_treatment_eq / unpriv_treatment_eq

    return all_fairness_scores

    def treatment_score(truth, predict):
        """
        Calculate treatement score (here using ratio of false neg to false pos)

        Args:
            truth (list): truth labels for the given data
            predict (list): predicted labels for the given data

        Returns:
            float: treatment score for the given data

        """
        fn_index = [1 for i in range(len(predict)) if truth[i] != predict[i] if truth[i] == 0]
        fp_index = [1 for i in range(len(predict)) if truth[i] != predict[i] if truth[i] == 1]
        return len(fn_index) / len(fp_index)
