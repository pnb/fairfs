# Using pyenv environment "fairfs" (Python 3.8)
import shap
import pandas as pd
import numpy as np
import math
from sklearn import metrics, model_selection, pipeline, preprocessing
from sklearn import tree

import dataset_loader
from column_threshold_selector import ColumnThresholdSelector


PROTECTED_COLUMN = 'Sex'  # 'Sex' for adult, 'group' for synthetic
PRIVILEGED_VALUE = 1      # 1 for synthetic and for adult (indicates male)
UNPRIVILEGED_VALUE = 0    # 0 for synthetic and for adult (indicates female)
ITERATIONS = 100
ACCURACY_METRIC = metrics.roc_auc_score
COLUMNS_SYNTH = ['group', 'fair_feature', 'unfair_feature']
COLUMNS_ADULT = ['Age', 'Workclass', 'Education-Num', 'Marital Status',
                 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain',
                 'Capital Loss', 'Hours per week', 'Country']
STAT_PARITY_COLUMNS_SYNTH = ['group 1', 'group 0', 'ratio']
STAT_PARITY_COLUMNS_ADULT = ['male', 'female', 'ratio']
FAIRNESS_METRICS_LIST = ['stat_parity', 'treatment_eq_ratio']
SELECTION_METRIC = 'stat_parity'  # change as desired
SELECTION_CUTOFFS = [.1, .2, .3, .4]
CUTOFF_VALUE = 0.2

model = tree.DecisionTreeClassifier(random_state=11798)


def main2():
    X, y = shap.datasets.adult()
    columns = X.columns

    # Create list to hold fairness and accuracy for each run
    results = []

    # Create 10-fold cross-validation train test split for the overall model
    cross_val = model_selection.KFold(10, shuffle=True, random_state=11798)

    for train_index, test_index in cross_val.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = pd.DataFrame(X_train, columns=columns)
        featureSelector = ColumnThresholdSelector(
            model, PRIVILEGED_VALUE, CUTOFF_VALUE, SELECTION_METRIC)
        featureSelector.fit(X_train, y_train)
        print(featureSelector.transform(X_test))


def main():
    # Simulated dataset
    # ds = dataset_loader.get_simulated_data()['simulated_data']
    # X, y = ds['data'], pd.Series(ds['labels'])

    # Adult dataset (comes pre-cleaned in shap library)
    X, y = shap.datasets.adult()
    columns = X.columns

    # Create list to hold fairness and accuracy for each run
    results = []

    # Create 10-fold cross-validation train test split for the overall model
    cross_val = model_selection.KFold(10, shuffle=True, random_state=11798)

    for train_index, test_index in cross_val.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = pd.DataFrame(X_train, columns=columns)

        # run initial model for feature selection on training data only
        accuracy, shap_values, pred_labels = run_model(X_train, y_train)
        fairness_values = calc_feature_fairness_scores(X_train, y, shap_values)

        selected_features = select_features(fairness_values, CUTOFF_VALUE)
        print("features: ", selected_features)

        # specify columns to include
        X_train_drop_features = pd.DataFrame(X_train, columns=selected_features)
        X_test = pd.DataFrame(X_test, columns=selected_features)

        # Run the model as defined in the constants, get predictions and accuracy
        model.fit(X_train_drop_features, y_train)
        predictions = model.predict(X_test)

        results.append(pd.Series({
            'model': model.__class__.__name__,
            'unfairness_metric': SELECTION_METRIC,
            'auc': ACCURACY_METRIC(y_test, predictions),
            'model_fairness': calc_overall_fairness_scores(X_test, y_test, predictions),
            'columns': selected_features,
            'cutoff': CUTOFF_VALUE
        }))

    pd.concat(results).to_csv('fairfs_shap_results.csv', index=False)


def run_model(data, labels):
    """ Runs training with 4-fold cross validation and returns
    a dataframe of accuracy (AUC) for each iteration, a list of shap values
    for each datapoint, and prediction labels for each data point.

    Args:
        data (DataFrame): data to be used in model
        labels (list): labels for data being used in model

    Returns:
        List: list of DataFrames (one per CV fold) with AUC accuracy
        DataFrame: shapley values for the model
        DataFrame: predicted labels for the model

    """
    # Create the DataFrame to hold the SHAP results, labels, and accuracy
    shap_values = pd.DataFrame(index=data.index, columns=data.columns)
    pred_labels = pd.DataFrame(index=data.index, columns=['labels'])
    accuracy = []

    # Create cross-validation train test split
    cross_val = model_selection.KFold(4, shuffle=True, random_state=11798)

    for train_index, test_index in cross_val.split(data, labels):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X_train = pd.DataFrame(X_train, columns=data.columns)
        X_test = pd.DataFrame(X_test, columns=data.columns)

        # Run the model as defined in the constants, get predictions and accuracy
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy.append(pd.Series({
            'label_type': 'y',
            'auc': ACCURACY_METRIC(y_test, predictions)
        }))

        # Get shap values
        explainer = shap.TreeExplainer(model)
        current_shap_values = explainer.shap_values(X_test)
        # Only need to save one end of the range of values
        shap_values.iloc[test_index] = current_shap_values[0]

        # Get predicted labels
        pred_labels.iloc[test_index] = predictions[0]

    return accuracy, shap_values, pred_labels


def select_features(columnwise_values, cutoff_value):
    """
    Take fairness values for each column and use the given metric to remove
    most unfair features using cutoff. "Most unfair" is measured by absolute distance
    from 1 (which represents the classes being exactly equal)

    Args:
        columnwise_values (DataFrame): DataFrame containing fairness calculation for each column

    Returns:
        Index: Pandas Index of columns to use

    """
    # calculate relative unfairness by getting distance of each value from 1
    adjusted_values = abs(columnwise_values - 1)

    # Get columns sorted for relevant metric (transposed to allow dropping)
    sorted_cols = adjusted_values.T.sort_values(by=[SELECTION_METRIC])

    # Get number of columns to drop
    cutoff_index = math.floor(CUTOFF_VALUE * len(sorted_cols))

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


def overall_accuracy(truth, predict):
    """
    Calculate overall accuracy for each group

    Args:
        truth (list): truth labels for the given data
        predict (list): predicted labels for the given data

    Returns:
        float: overall percentage of true pos and negs for the given data and model

    """
    tp_tn_index = truth == predict
    return (np.count_nonzero(tp_tn_index)) / len(predict)


def marginal_dist(predict):
    """
    Calculate marginal distribution

    Args:
        predict (list): predicted labels for the given data

    Returns:
        float: marginal distribution score for the given data

    """
    return (np.count_nonzero(predict)) / len(predict)


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


def calc_feature_fairness_scores(data, labels, shap_values):
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
    all_fairness_scores = pd.DataFrame(index=FAIRNESS_METRICS_LIST, columns=cols)
    converted_df = convert_shap_to_bools(data, shap_values)

    priv_indices = (pd.DataFrame((
        [data.loc[data[PROTECTED_COLUMN] == PRIVILEGED_VALUE]])[0])).index
    priv_truth = labels[priv_indices]

    unpriv_indices = (pd.DataFrame((
        [data.loc[data[PROTECTED_COLUMN] == UNPRIVILEGED_VALUE]])[0])).index
    unpriv_truth = labels[unpriv_indices]

    for col in cols:
        # Get predictions for privileged and unprivileged classes
        priv_predict = converted_df[col][priv_indices].tolist()
        unpriv_predict = converted_df[col][unpriv_indices].tolist()

        # Get statistical parity (ratio of marginal distributions)
        # close to 1 means marginal distributions are equal
        # less than 1 means priv group has fewer predicted pos than unpriv
        # greater than 1 means priv group has more predicted pos than unpriv
        priv_marg_dist = marginal_dist(priv_predict)
        unpriv_marg_dist = marginal_dist(unpriv_predict)
        all_fairness_scores[col]['stat_parity'] = priv_marg_dist / unpriv_marg_dist

        # Get treatment score, stored as ratio for comparison purposes
        # greater than 1 means unpriv group has greater false neg to false pos ratio than priv group
        # 1 is equal rates of false neg to false pos for both groups
        # less than 1 means priv group has greater false neg to false pos ratio than unpriv group
        priv_treatment_eq = treatment_score(priv_truth, priv_predict)
        unpriv_treatment_eq = treatment_score(unpriv_truth, unpriv_predict)
        all_fairness_scores[col]['treatment_eq_ratio'] = priv_treatment_eq / unpriv_treatment_eq

    return all_fairness_scores


def calc_overall_fairness_scores(data, truth, predict):
    """
    Calculate fairness scores for overall model.

    Args:
        data (DataFrame): original dataset used for train and test
        truth (Array): list of truth labels for the data
        predict (Array): list of predicted labels for the data
    Returns:
        DataFrame: fairness scores for the model

    """
    fairness_scores = pd.DataFrame()
    predictions_df = pd.DataFrame(truth, columns=['labels'], index=data.index)

    priv_indices = (pd.DataFrame((
        [data.loc[data[PROTECTED_COLUMN] == PRIVILEGED_VALUE]])[0])).index
    priv_truth = truth[priv_indices]

    unpriv_indices = (pd.DataFrame((
        [data.loc[data[PROTECTED_COLUMN] == UNPRIVILEGED_VALUE]])[0])).index
    unpriv_truth = truth[unpriv_indices]

    priv_predict = predictions_df['labels'][priv_indices].tolist()
    unpriv_predict = predictions_df['labels'][unpriv_indices].tolist()

    # Get statistical parity (ratio of marginal distributions)
    # close to 1 means marginal distributions are equal
    # less than 1 means priv group has fewer predicted pos than unpriv
    # greater than 1 means priv group has more predicted pos than unpriv
    priv_marg_dist = marginal_dist(priv_predict)
    unpriv_marg_dist = marginal_dist(unpriv_predict)
    fairness_scores['stat_parity'] = [priv_marg_dist / unpriv_marg_dist]

    # Get treatment score, stored as ratio for comparison purposes
    # greater than 1 means unpriv group has greater false neg to false pos ratio than priv group
    # 1 is equal rates of false neg to false pos for both groups
    # less than 1 means priv group has greater false neg to false pos ratio than unpriv group
    priv_treatment_eq = treatment_score(priv_truth, priv_predict)
    unpriv_treatment_eq = treatment_score(unpriv_truth, unpriv_predict)
    fairness_scores['treatment_eq_ratio'] = [priv_treatment_eq / unpriv_treatment_eq]

    return(fairness_scores)


if __name__ == "__main__":
    main2()
