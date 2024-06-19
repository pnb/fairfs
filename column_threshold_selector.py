# Object for selecting columns from dataset using threshold in scikit-learn pipelines.
import pandas as pd
import math
import numpy as np
import shap
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import shuffle
import unfairness_metrics

class ColumnThresholdSelector(BaseEstimator, TransformerMixin):
    """
    Object for selecting columns that contribute least to unfairness from a dataset given a threshold.
    The provided threshold indicates the percentage of features to keep.

    Parameters
    ----------
        columnwise_values (DataFrame): DataFrame containing fairness calculation for each column

    Returns
    ----------
        Index: Pandas Index of columns to use

    """

    def __init__(self, estimator: object, group_membership: pd.Series, cutoff_value: float,
                 unfairness_metric: object, rand_seed: int = 42, sample_groupings: pd.Series = None):
        """

        Parameters
        ----------
        estimator : scikit-learn estimator object
            The estimator being used for the prediction task.
        group_membership : pd.Series
            Boolean values referring to whether a given row is a member of the privileged class.
        cutoff_value : float
            The percentage of features, expressed as a decimal, to select for use in the prediction task.
            e.g., 0.8 means 80% of the features will be kept.
            This is rounded down to the nearest whole number when selecting features.
        unfairness_metric : UnfairnessMetric object
            Unfairness metric used to compare the impact of features on unfairness in the predictions.
        rand_seed : int (default 42)
            Integer to seed random number generator.
            TO DO: document sample_groupings
        """
        self.estimator = estimator
        self.group_membership = group_membership
        self.cutoff_value = cutoff_value
        self.unfairness_metric = unfairness_metric
        self.selected_features = []
        self.rand_seed = rand_seed
        self.sample_groupings = sample_groupings

    def fit(self, X: pd.DataFrame, y: pd.Series):
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
            fairness_values = self.full_cv_fit(X, y)
            # TODO is this different if there are groups and fewer than 500 datapoints?

        # otherwise get split of training and testing data randomly, where test data is
        # small subset for speed
        else:
            if self.sample_groupings is not None:
                all_pids = shuffle(self.sample_groupings.unique())
                # breakpoint()
                X_train = pd.DataFrame()
                y_train = pd.Series()
                while len(X_train) < 250:
                    # need breakpoint here
                    # breakpoint()
                    cur_pid = all_pids[0]
                    all_pids = all_pids[1:]
                    cur_X_pid_rows = X.loc[self.sample_groupings == cur_pid]
                    cur_y_pid_rows = y.loc[self.sample_groupings == cur_pid]
                    X_train = pd.concat([X_train, cur_X_pid_rows])
                    y_train = pd.concat([y_train, cur_y_pid_rows])

                
                # X_train = pd.concat(X_train)  # might be more than 250, needed because we don't want data leakage from train to test
                # y_train = pd.concat(y_train)
                assert X_train.index.equals(y_train.index),  "Training data and labels do not have matching indices"
                X_test = X.drop(index=X_train.index)
                y_test = y.drop(index=y_train.index)

            else:
                X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=250,
                                                                                random_state=self.rand_seed)
            

            # Run the model as defined in the constants, get predictions and accuracy
            cloned_estimator = clone(self.estimator)
            cloned_estimator.fit(X_train, y_train)

            shap_values = self.fit_explainer(cloned_estimator, X_train, X_test)

            # feature selection
            fairness_values = self.calc_feature_unfairness_scores(X_test,
                                                                  y_test,
                                                                  shap_values)

        # select features
        self.selected_features = self.select_features(fairness_values, self.cutoff_value)

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """ return data with only fair features included. y is optional and unused

        Parameters
        ----------
        X : Pandas DataFrame of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y

        Returns
        -------
        DataFrame of shape (n_samples, m_features)
            Training vector, where 'n_samples' is still the number of samples but m_features has
            been trimmed to select the features that contribute to unfairness the least.
        """
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y=None):
        """

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
            Fitted and transformed estimator.
        """
        # call fit and then call transform
        return self.fit(X, y).transform(X)

    def full_cv_fit(self, X: pd.DataFrame, y):
        shap_vals = pd.DataFrame(index=X.index, columns=X.columns)
        cross_val = model_selection.KFold(4, shuffle=True, random_state=11798)
        y = np.array(y)

        for train_index, test_index in cross_val.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_test = pd.DataFrame(X_test, columns=X.columns)

            # Run the model as defined in the constants, get predictions and accuracy
            cloned_estimator = clone(self.estimator)
            cloned_estimator.fit(X_train, y_train)

            # Get shap values
            shap_vals.iloc[test_index] = self.fit_explainer(cloned_estimator, X_train, X_test)

        # feature selection
        fairness_values = self.calc_feature_unfairness_scores(X,
                                                              y,
                                                              shap_vals)
        return fairness_values

    def fit_explainer(self, estimator, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """

        Parameters
        ----------
        estimator : scikit-learn estimator object
        X_train : Pandas DataFrame of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        X_test : Pandas DataFrame of shape (m_samples, m_features)
            Test vector, where `m_samples` is the number of samples and
            `m_features` is the number of features.

        Returns
        -------
        Dataframe of shap values with size and shape equal to X_test (m_samples and m_features)

        """
        try:
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(X_test)[0]
        except shap.utils._exceptions.InvalidModelError:
            try:
                explainer = shap.LinearExplainer(estimator, X_train)
                values = explainer.shap_values(X_test)
            except shap.utils._exceptions.InvalidModelError:
                # send in X_train sample and random seed to the explainer rather than the entire test dataset
                explainer = shap.KernelExplainer(
                    estimator.predict,
                    X_train,
                    keep_index=True
                )
                values = explainer.shap_values(X_test)

        return pd.DataFrame(columns=X_test.columns, index=X_test.index, data=values)

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
        assert (len(feature_unfairness_scores) * cutoff_value) > 1, \
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
                labels, shap_values, self.group_membership[data.index], self.unfairness_metric)
            feature_unfairness_scores[col] = unfairness_score

        return feature_unfairness_scores
