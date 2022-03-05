# Using pyenv environment "fairfs" (Python 3.8)
import shap
import pandas as pd
import numpy as np
from tqdm import tqdm
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import metrics, model_selection, pipeline, preprocessing
from sklearn import tree

import dataset_loader


PROTECTED_COLUMN = 'Sex'  # 'Sex' for adult, 'group' for synthetic
PRIVILEGED_VALUE = 1      # 1 for synthetic and for adult (indicates male)
UNPRIVILEGED_VALUE = 0    # 0 for synthetic and for adult (indicates female)
ITERATIONS = 100
ACCURACY_METRIC = metrics.roc_auc_score
MODEL = tree.DecisionTreeClassifier(random_state=11798)
COLUMNS_SYNTH = ['group', 'fair_feature', 'unfair_feature']
COLUMNS_ADULT = ['Age', 'Workclass', 'Education-Num', 'Marital Status',
                 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain',
                 'Capital Loss', 'Hours per week', 'Country']
STAT_PARITY_COLUMNS_SYNTH = ['group 1', 'group 0', 'ratio']
STAT_PARITY_COLUMNS_ADULT = ['male', 'female', 'ratio']
FAIRNESS_METRICS_LIST = ['overall_accuracy', 'stat_parity', 'treatment_eq_ratio']
SELECTION_METRIC = 'stat_parity' # change as desired


def main():
    # Adult dataset (comes pre-cleaned in shap library)
    X, y = shap.datasets.adult()
    columns = X.columns

    # Simulated dataset
    # ds = dataset_loader.get_simulated_data()['simulated_data']
    # X, y = ds['data'], pd.Series(ds['labels'])

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
        fairness_values = calc_feature_fairness_scores(X_train, y_train, shap_values)

        selected_features = select_features(fairness_values)

        # specificy columns to include
        X_train_new = pd.DataFrame(X_train, columns=selected_features)
        X_test = pd.DataFrame(X_test, columns=selected_features)

        # Run the model as defined in the constants, get predictions and accuracy
        model = MODEL.fit(X_train_new, y_train)
        predictions = model.predict(X_test)
        results.append(pd.Series({
            'model': MODEL,
            'unfairness_metric': SELECTION_METRIC,
            'auc': ACCURACY_METRIC(y_test, predictions),
            'columns': selected_features,
        }))

    # shap_values.to_csv('fairfs_shap_results1.csv', index=False, encoding='utf-8')
    # pred_labels.to_csv('fairfs_shap_labels1.csv', index=False, encoding='utf-8')
    # fairness_values.to_csv('fairfs_fairness_values1.csv', index=True, encoding='utf-8')


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
    shap_vals = pd.DataFrame(index=data.index, columns=data.columns)
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
        model = MODEL.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy.append(pd.Series({
            'label_type': 'y',
            'auc': ACCURACY_METRIC(y_test, predictions)
        }))

        # Get shap values
        explainer = shap.TreeExplainer(model)
        current_shap_values = explainer.shap_values(X_test)
        # Only need to save one end of the range of values
        shap_vals.iloc[test_index] = current_shap_values[0]

        # Get predicted labels
        pred_labels.iloc[test_index] = predictions[0]

    return accuracy, shap_vals, pred_labels


def select_features(columnwise_values):
    """
    Take fairness values for each column and use the given metric to remove
    most unfair features using cutoff.

    Args:
        columnwise_values (DataFrame): DataFrame containing fairness calculation for each column

    Returns:
        Series: Series object of columns to use

    """
    col_values = columnwise_values.loc[SELECTION_METRIC]
    # to-do: add selection mechanism
    return col_values


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
        priv_predict = converted_df[col].iloc[priv_indices].tolist()
        unpriv_predict = converted_df[col].iloc[unpriv_indices].tolist()

        # Get overall accuracy ratio
        # close to 1 means equally accurate for both groups
        # greater than 1 means priv group has more accurate predictions
        # less than 1 means unpriv group has more accurate predictions
        priv_accuracy = overall_accuracy(priv_truth, priv_predict)
        unpriv_accuracy = overall_accuracy(unpriv_truth, unpriv_predict)
        all_fairness_scores[col]['stat_parity'] = priv_accuracy / unpriv_accuracy

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


if __name__ == "__main__":
    main()
