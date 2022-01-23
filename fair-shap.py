# Using pyenv environment "fairfs" (Python 3.8)
import shap
import pandas as pd
import numpy as np
from tqdm import tqdm
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import metrics, model_selection, pipeline, preprocessing
from sklearn import tree

import dataset_loader
import unfairness_metrics


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

    Keyword arguments:
    data: pandas dataframe of data to be used in model
    labels: list of labels for data
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
    """ Convert all values to 0 or 1 where 0 is a neg shap value and 1 is a
    pos shap value
    """
    converted_df = pd.DataFrame(index=data.index, columns=data.columns)
    for column in shap_values:
        values = []
        for value in shap_values[column]:
            values.append(0 if value < 0 else 1)

        converted_df[column] = values

    return converted_df


def marginal_dist(truth, predict):
    """ Calculate marginal distribution (percentage of cases where truth equals
    prediction)
    """
    tp_tn_index = truth == predict
    return (np.count_nonzero(tp_tn_index)) / len(predict)


def treatment_score(truth, predict):
    """ Calculate treatement score (here using ration of false neg to false pos)"""
    fn_index = [1 for i in range(len(predict)) if truth[i] != predict[i] if truth[i] == 0]
    fp_index = [1 for i in range(len(predict)) if truth[i] != predict[i] if truth[i] == 1]
    return len(fn_index) / len(fp_index)


def stat_parity(data, labels, cols, shap_values):
    """ Calculate marginal distribution for priv and unpriv group, then find ratio.
    Return a dataframe with all three values per column (feature)
    """
    all_marginal_dist = pd.DataFrame(index=STAT_PARITY_COLUMNS_ADULT, columns=cols)
    converted_df = convert_shap_to_bools(data, shap_values)

    privileged_indices = (pd.DataFrame((
        [data.loc[X[PROTECTED_COLUMN] == PRIVILEGED_VALUE]])[0])).index
    privileged_labels = labels[privileged_indices]

    unprivileged_indices = (pd.DataFrame((
        [data.loc[X[PROTECTED_COLUMN] == UNPRIVILEGED_VALUE]])[0])).index
    unprivileged_labels = labels[unprivileged_indices]

    for col in cols:
        all_marginal_dist[col]['male'] = marginal_dist(privileged_labels,
                                                       converted_df[col].
                                                       iloc[privileged_indices].
                                                       tolist())
        all_marginal_dist[col]['female'] = marginal_dist(unprivileged_labels,
                                                         converted_df[col].
                                                         iloc[unprivileged_indices].
                                                         tolist())
        all_marginal_dist[col]['ratio'] = all_marginal_dist[col]['male'] / all_marginal_dist[col]['female']

    return all_marginal_dist


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
