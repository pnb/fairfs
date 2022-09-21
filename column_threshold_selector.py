# Object for selecting columns from dataset using threshold in scikit-learn pipelines.
import pandas as pd
import np
import math
import shap
from sklearn import model_selection
from sklearn.base import BaseEstimator, TransformerMixin
import unfairness_metrics


class ColumnThresholdSelector(BaseEstimator, TransformerMixin):
    # to do: update doc string
    """Object for selecting columns from a dataset given a threshold.

    Parameters:
        columnwise_values (DataFrame): DataFrame containing fairness calculation for each column

    Returns:
        Index: Pandas Index of columns to use

    """

    def __init__(self, model, group_membership, privileged_value, cutoff_value, unfairness_metric):
        self.model = model
        self.group_membership = group_membership
        self.privileged_value = privileged_value
        self.cutoff_value = cutoff_value
        self.unfairness_metric = unfairness_metric
        self.selected_features = []

    def fit(self, X, y):
        """ Actual fitting of model,
        choosing features given parameters, etc
        mostly will be moved over from run_model in fair_shap.py
        """
        assert isinstance(X, pd.DataFrame), 'Only pd.DataFrame inputs for X are supported'
        # Create the DataFrame to hold the SHAP results, labels, and accuracy
        shap_values = pd.DataFrame(index=X.index, columns=X.columns)

        # TODO: get split of training and testing data randomly, where test is much smaller
        #np.random_sample

        # Create cross-validation train test split
        # cross_val = model_selection.KFold(4, shuffle=True, random_state=11798)
        #
        # train_index, test_index = cross_val.split(X, y)[0]

# commenting for speed
        # for train_index, test_index in cross_val.split(X, y):
        #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #
        #     X_train = pd.DataFrame(X_train, columns=X.columns)
        #     X_test = pd.DataFrame(X_test, columns=X.columns)
        #
        #     # Run the model as defined in the constants, get predictions and accuracy
        #     self.model.fit(X_train, y_train)
        #     predictions = self.model.predict(X_test)
        #
        #     # Get shap values
        #     explainer = shap.TreeExplainer(self.model)
        #     current_shap_values = explainer.shap_values(X_test)
        #     # Only need to save one end of the range of values
        #     shap_values.iloc[test_index] = current_shap_values[0]

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

    def select_features(self, feature_unfairness_scores, cutoff_value):
        """Take unfairness values for each column and use the given metric to remove.
        most unfair features using cutoff. Larger values are more unfair, and unfairness
        values range from 0 to 1

        Args:
            columnwise_values (DataFrame): DataFrame containing fairness calculation for each column
            cutoff_value (Float): Percentage of columns to keep
        Returns:
            Series: Pandas Series of features to use

        """
        # Sort unfairness values in ascending order, returns list of scores
        sorted_cols = feature_unfairness_scores.sort_values()

        # Get number of columns to drop
        cutoff_index = math.floor(self.cutoff_value * len(sorted_cols))

        # select only desired columns by cutting off highest values
        new_cols = sorted_cols.iloc[:cutoff_index]

        # return Series with only selected features
        return new_cols.index

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
        converted_df = pd.DataFrame(index=shap_values.index, columns=shap_values.columns)
        for column in shap_values:
            values = []
            for value in shap_values[column]:
                values.append(0 if value < 0 else 1)

            converted_df[column] = values

        return converted_df

    def calc_feature_unfairness_scores(self, data, labels, shap_values):
        """
        Calculate unfairness scores for the existing metrics and store for each column.

        Args:
            data (DataFrame): original dataset used for train and test
            labels (Array): list of truth labels for the data
            shap_values (DataFrame): shapley labels for the model
        Returns:
            Series: unfairness scores for each column

        """
        cols = data.columns
        feature_unfairness_scores = pd.Series(index=cols, dtype=float)  # convert to series or dict
        converted_df = self.convert_shap_to_bools(data, shap_values)

        # asserts that the length of the test values are equal
        assert len(labels) == len(
            shap_values), "Error: length of labels is not equal to the length of shap values"
        assert len(data) == len(labels), "Error: length of data is not equal to length of labels"

        for col in cols:
            shap_values = converted_df[col]
            # group_membership = self.group_membership
            unfairness_score = unfairness_metrics.calc_unfairness(
                labels, shap_values, self.group_membership, self.unfairness_metric)
            feature_unfairness_scores[col] = unfairness_score

        return feature_unfairness_scores
