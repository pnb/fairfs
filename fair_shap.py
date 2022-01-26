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


# Get average of values in dictionary
def average_val(dict):
    avgs = {}
    for key, val in dict.items():
        avgs[key] = np.mean(val)
    return avgs


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
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
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


def convert_shap_to_bools(data, shap_values):
    """
    Convert shapley values to booleans. If shapley value is less than 0, then
    set to 0. If greater than or equal to 0, set to 1. This allows for
    traditional fairness metrics to be applied to shapley values.

    Args:
        data (DataFrame): original dataset used for train and test
        shap_values (DataFrame): shapley values for the model

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


def marginal_dist(truth, predict):
    """
    Calculate marginal distribution

    Args:
        truth (list): truth labels for the given data
        predict (list): predicted labels for the given data

    Returns:
        float: marginal distribution score for the given data

    """
    tp_tn_index = truth == predict
    return (np.count_nonzero(tp_tn_index)) / len(predict)


def treatment_score(truth, predict):
    """
    Calculate treatement score (here using ration of false neg to false pos)

    Args:
        truth (list): truth labels for the given data
        predict (list): predicted labels for the given data

    Returns:
        float: treatment score for the given data

    """
    fn_index = [1 for i in range(len(predict)) if truth[i] != predict[i] if truth[i] == 0]
    fp_index = [1 for i in range(len(predict)) if truth[i] != predict[i] if truth[i] == 1]
    return len(fn_index) / len(fp_index)


def calc_fairness_scores(data, labels, shap_values):
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
    all_fairness_scores = [pd.DataFrame(index=data.index, columns=cols)]
    converted_df = convert_shap_to_bools(data, shap_values)

    priv_indices = (pd.DataFrame((
        [data.loc[X[PROTECTED_COLUMN] == PRIVILEGED_VALUE]])[0])).index
    priv_truth = labels[priv_indices]

    unpriv_indices = (pd.DataFrame((
        [data.loc[X[PROTECTED_COLUMN] == UNPRIVILEGED_VALUE]])[0])).index
    unpriv_truth = labels[unpriv_indices]

    for col in cols:
        # Get predictions for privileged and unprivileged classes
        priv_predict = converted_df[col].iloc[priv_indices].tolist()
        unpriv_predict = converted_df[col].iloc[unpriv_indices].tolist()

        # Get statistical parity (ratio of marginal distributions)
        priv_marg_dist = marginal_dist(priv_truth, priv_predict)
        unpriv_marg_dist = marginal_dist(unpriv_truth, unpriv_predict)
        all_fairness_scores[col]['stat_parity'] = priv_marg_dist / unpriv_marg_dist

        # Get treatment score, stored as ratio for comparison purposes
        priv_treatment_eq = treatment_score(priv_truth, priv_predict)
        unpriv_treatment_eq = treatment_score(unpriv_truth, unpriv_predict)
        all_fairness_scores[col]['treatment_eq_ratio'] = priv_treatment_eq / unpriv_treatment_eq

    return all_fairness_scores


if __name__ == "__main__":
    # Adult dataset (comes pre-cleaned in shap library)
    X, y = shap.datasets.adult()
    columns = X.columns

    # Simulated dataset
    # ds = dataset_loader.get_simulated_data()['simulated_data']
    # X, y = ds['data'], pd.Series(ds['labels'])

    accuracy, shap_values, pred_labels = run_model(X, y)

    # results = pd.DataFrame({
    #     'shap_values': shap_values,
    #     'pred_labels': pred_labels
    # })

    # accuracy.to_csv('fairfs_shap_accuracy.csv', index=False, encoding='utf-8') #to do: fix
    shap_values.to_csv('fairfs_shap_results.csv', index=False, encoding='utf-8')
    pred_labels.to_csv('fairfs_shap_labels.csv', index=False, encoding='utf-8')
