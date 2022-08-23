# Object for selecting columns from dataset using threshold in scikit-learn pipelines.
import pandas as pd
import numpy as np
import math
import shap
from sklearn import metrics, model_selection, pipeline, preprocessing
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin
import unfairness_metrics


class ColumnThresholdSelector(BaseEstimator, TransformerMixin):
    """Object for selecting columns from a dataset given a threshold.

    Parameters:
        columnwise_values (DataFrame): DataFrame containing fairness calculation for each column

    Returns:
        Index: Pandas Index of columns to use

    """

    def __init__(self, model, sensitive_column, privileged_value, cutoff_value, unfairness_metric):
        self.model = model
        self.sensitive_column = sensitive_column
        self.privileged_value = privileged_value
        self.cutoff_value = cutoff_value
        # string or function. if string match to existing function or people can pass in their own function
        self.unfairness_metric = unfairness_metric
        self.selected_features = []

    def fit(self, X, y):
        """ actual fitting of model,
        choosing features given parameters, etc
        mostly will be moved over from run_model in fair_shap.py
        """
        assert isinstance(X, pd.DataFrame), 'Only pd.DataFrame inputs for X are supported'
        # Create the DataFrame to hold the SHAP results, labels, and accuracy
        shap_values = pd.DataFrame(index=X.index, columns=X.columns)
        # pred_labels = pd.DataFrame(index=X.index, columns=['labels'])

        # Create cross-validation train test split
        cross_val = model_selection.KFold(4, shuffle=True, random_state=11798)

        for train_index, test_index in cross_val.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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
            shap_values.iloc[test_index] = current_shap_values[0]

            # Get predicted labels
            # pred_labels.iloc[test_index] = predictions[0]

        # feature selection
        fairness_values = self.calc_feature_unfairness_scores(X,
                                                              y,
                                                              shap_values)

        # select features
        self.selected_features = self.select_features(fairness_values, self.cutoff_value)

        return self

    def transform(self, X, y=None):
        """ return data with only fair features included. y is optional for unsupervised models
        """
        return X[self.selected_features]

    def fit_transform(self, X, y=None):
        #call fit and then call transform
        return self.fit(X, y).transform(X)

    def select_features(self, columnwise_values, cutoff_value):
        """Take unfairness values for each column and use the given metric to remove.
        most unfair features using cutoff. Larger values are more unfair, and unfairness
        values range from 0 to 1

        Args:
            columnwise_values (DataFrame): DataFrame containing fairness calculation for each column
            cutoff_value (Float): Percentage of columns to keep
        Returns:
            Index: Pandas Index of columns to use

        """
        # Get columns sorted for relevant metric (transposed to allow dropping)
        sorted_cols = adjusted_values.T.sort_values(by=[self.unfairness_metric])

        # Get number of columns to drop
        cutoff_index = math.floor(self.cutoff_value * len(sorted_cols))

        # select only desired columns by cutting off highest value rows
        new_cols = sorted_cols.iloc[:-cutoff_index]

        # return list of columns as Index object
        return new_cols.T.columns

    def convert_shap_to_bools(self, data, shap_values):
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
        # print(data)
        # print(shap_values)
        converted_df = pd.DataFrame(index=shap_values.index, columns=shap_values.columns)
        # print(converted_df)
        for column in shap_values:
            values = []
            for value in shap_values[column]:
                values.append(0 if value < 0 else 1)

            converted_df[column] = values

        return converted_df

    #to-do: rename function
    def calc_feature_unfairness_scores(self, data, labels, shap_values):
        """
        Calculate unfairness scores for the existing metrics and store for each column.

        Args:
            data (DataFrame): original dataset used for train and test
            labels (Array): list of truth labels for the data
            shap_values (DataFrame): shapley labels for the model
        Returns:
            DataFrame: unfairness scores for each column

        """
        cols = data.columns
        all_unfairness_scores = pd.DataFrame(index=[self.unfairness_metric], columns=cols)
        converted_df = self.convert_shap_to_bools(data, shap_values)
        print(len(labels), len(converted_df), len(data))
        assert len(labels) == len(shap_values), "failed labels and shap"
        assert len(data) == len(labels), "failed data and labels"

        for col in cols:

            unfairness_score = unfairness_metrics.calc_unfairness(
                labels, converted_df[col], data[self.sensitive_column], self.unfairness_metric)
            all_unfairness_scores[col][self.unfairness_metric] = unfairness_score

        return all_unfairness_scores

# def treatment_score(truth, predict):
#     """
#     Calculate treatement score (here using ratio of false neg to false pos)
#
#     Args:
#     truth (list): truth labels for the given data
#     predict (list): predicted labels for the given data
#
#     Returns:
#     float: treatment score for the given data
#
#     """
#     fn_index=[1 for i in range(len(predict)) if truth[i] != predict[i] if truth[i] == 0]
#     fp_index=[1 for i in range(len(predict)) if truth[i] != predict[i] if truth[i] == 1]
#     return len(fn_index) / len(fp_index)
#
#
# def marginal_dist(predict):
#     """
#     Calculate marginal distribution
#
#     Args:
#     predict (list): predicted labels for the given data
#
#     Returns:
#     float: marginal distribution score for the given data
#
#     """
#     return (np.count_nonzero(predict)) / len(predict)
