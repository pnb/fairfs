# Object for selecting columns from dataset using threshold in scikit-learn pipelines.
import pandas as pd
import math
import shap
from sklearn import model_selection
from sklearn.base import BaseEstimator, TransformerMixin, clone
import unfairness_metrics


class ColumnThresholdSelector(BaseEstimator, TransformerMixin):
    # to do: update doc string
    """Object for selecting columns from a dataset given a threshold.

    Parameters:
        columnwise_values (DataFrame): DataFrame containing fairness calculation for each column

    Returns:
        Index: Pandas Index of columns to use

    """

    def __init__(self, estimator, group_membership, cutoff_value, unfairness_metric, rand_seed=42):
        self.estimator = estimator
        self.group_membership = group_membership
        self.cutoff_value = cutoff_value
        self.unfairness_metric = unfairness_metric
        self.selected_features = []
        self.rand_seed = rand_seed

    def fit(self, X: pd.DataFrame, y):
        """ Actual fitting of model,
        choosing features given parameters, etc

        Parameters
        ----------
        X : Pandas DataFrame of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y: array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        assert isinstance(X, pd.DataFrame), 'Only pd.DataFrame inputs for X are supported'

        # if dataset contains fewer than 500 rows, do a full cross-validation
        if len(X.index) < 500:
            fairness_values = self.estimator.full_cv_fit(X, y)

        # otherwise get split of training and testing data randomly, where test data is
        # small subset for speed
        else:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=250, random_state=self.rand_seed)

            # Run the model as defined in the constants, get predictions and accuracy
            cloned_estimator = clone(self.estimator)
            cloned_estimator.fit(X_train, y_train)

            # Get shap values
            try:
                explainer = shap.TreeExplainer(cloned_estimator)

            except Exception:
                print("Tree Explainer will not work")
                exit()

            shap_values = pd.DataFrame(columns=X_test.columns, index=X_test.index, data=explainer.shap_values(X_test)[0])

            # feature selection
            fairness_values = self.calc_feature_unfairness_scores(X_test,
                                                                  y_test,
                                                                  shap_values)

        # select features
        self.selected_features = self.select_features(fairness_values, self.cutoff_value)

        return self

    def transform(self, X, y=None):
        """ return data with only fair features included. y is optional for unsupervised models
        """
        return X[self.selected_features]

    def fit_transform(self, X, y=None):
        # call fit and then call transform
        return self.fit(X, y).transform(X)

    def full_cv_fit(self, X, y):
        shap_vals = pd.DataFrame(index=X.index, columns=X.columns)
        cross_val = model_selection.KFold(4, shuffle=True, random_state=11798)

        for train_index, test_index in cross_val.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_test = pd.DataFrame(X_test, columns=X.columns)

            # Run the model as defined in the constants, get predictions and accuracy
            cloned_estimator = clone(self.estimator)
            cloned_estimator.fit(X_train, y_train)

            # Get shap values
            explainer = shap.TreeExplainer(cloned_estimator)
            current_shap_values = explainer.shap_values(X_test)
            # Only need to save one end of the range of values
            shap_vals.iloc[test_index] = current_shap_values[0]

        # feature selection
        fairness_values = self.calc_feature_unfairness_scores(X,
                                                              y,
                                                              shap_vals)
        return fairness_values

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
        # assertion that greater than 0 features will be kept
        assert (len(feature_unfairness_scores) * cutoff_value) < 1, \
            "Error: The provided cut-off is too small for the number of features"


        # Sort unfairness values in ascending order, returns list of scores
        # Uses numpy's underlying stable sort
        # https://numpy.org/doc/stable/reference/generated/numpy.sort.html#numpy.sort
        # If there is a tie, users can order the features by importance before running
        # TODO: check for ties and share message if so?
        sorted_cols = feature_unfairness_scores.sort_values(kind='stable')

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
            unfairness_score = unfairness_metrics.calc_unfairness(
                labels, shap_values, self.group_membership, self.unfairness_metric)
            feature_unfairness_scores[col] = unfairness_score

        return feature_unfairness_scores
