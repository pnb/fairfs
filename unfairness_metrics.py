# All metrics should return values in terms of unfairness, so that a higher value is worse. The
# returned value should have a minimum possible value of 0 and a maximum of 1.
# This paper (page 13) has 6 definitions: https://arxiv.org/pdf/1703.09207.pdf
# This paper discusses historical definitions: https://doi.org/10.1145/3287560.3287600
import pandas as pd
from sklearn import metrics
import numpy as np
from collections import defaultdict
from statistics import mean


UNFAIRNESS_METRICS = ['overall_accuracy_equality', 'statistical_parity', 'conditional_procedure',
                      'conditional_use_accuracy_equality', 'treatment_equality', 'all_equality']


def stat_score(truth, predict):
    return np.count_nonzero(predict)/len(predict)


def sigmoid(x):
    return (1 / (1 + np.exp(-x+1)))-0.5


def treatment_score(truth, predict):
    """
    when one of the number of false positive and false negative is 0, the other is not 0t,
    directly set treatment_score as 1.
    when both of the number of false positive and false negative are 0, directly set treatment_score as 0.
    otherwise, using sigmoid function to get the treatment_score.
    """
    fp_fn_index = truth != predict
    fp_fn_sum = np.count_nonzero(fp_fn_index)
    fp_fn = predict[fp_fn_index]
    fp_sum = np.count_nonzero(fp_fn)
    fn_sum = (fp_fn_sum-fp_sum)
    if fp_sum == fn_sum:
        return 0
    elif [fp_sum, fn_sum].count(0) == 1:
        return 1
    else:
        return sigmoid(max(fp_sum/fn_sum, fn_sum/fp_sum))


def calc_unfairness(y_true, y_pred, protected_groups, unfairness_metric):
    """Overall accuracy equality metric. Measures the maximum difference between accuracy
    (proportion correct) for each protected group. From https://arxiv.org/pdf/1703.09207.pdf, though
    note that this version allows for more than 2 classes and more than 2 protected groups. Multiple
    protected groups are handled by taking the maximum difference between each pair of protected
    groups as the unfairness measure, under the (arguable) assumption that people are most concerned
    with the worst-case scenario.

    Args:
        y_true (array-like): True classification labels, represented as integers
        y_pred (array-like): Predicted classification labels, represented as integers
        protected_groups (pd.Series): Group membership of each person/instance

    Returns:
        float: Unfairness measure in [0, 1], where 0 means perfect fairness
    """
    measurements = defaultdict(list)
    for group in protected_groups.unique():
        group_mask = protected_groups == group
        if unfairness_metric == 'overall_accuracy_equality':
            measurement = metrics.accuracy_score(y_true[group_mask], y_pred[group_mask])
        elif unfairness_metric == 'statistical_parity':
            measurement = stat_score(y_true[group_mask], y_pred[group_mask])
        elif unfairness_metric == 'conditional_procedure':
            measurement = metrics.recall_score(y_true[group_mask], y_pred[group_mask],
                                               zero_division=0)
        elif unfairness_metric == 'conditional_use_accuracy_equality':
            measurement = metrics.precision_score(y_true[group_mask], y_pred[group_mask],
                                                  zero_division=0)
        elif unfairness_metric == 'treatment_equality':
            measurement = treatment_score(y_true[group_mask], y_pred[group_mask])
        else:
            measurement = metrics.accuracy_score(y_true[group_mask], y_pred[group_mask])
            measurements['overall_accuracy_equality'].append(measurement)
            measurement = stat_score(y_true[group_mask], y_pred[group_mask])
            measurements['statistical_parity'].append(measurement)
            measurement = metrics.recall_score(y_true[group_mask], y_pred[group_mask],
                                               zero_division=0)
            measurements['conditional_procedure'].append(measurement)
            measurement = metrics.precision_score(y_true[group_mask], y_pred[group_mask],
                                                  zero_division=0)
            measurements['conditional_use_accuracy_equality'].append(measurement)
            measurement = treatment_score(y_true[group_mask], y_pred[group_mask])
            measurements['treatment_equality'].append(measurement)
            continue
        measurements[unfairness_metric].append(measurement)
    measurements[unfairness_metric] = [max(measurements[i]) - min(measurements[i])
                                       for i in measurements]
    if unfairness_metric == 'all_equality':
        return mean(measurements[unfairness_metric])
    else:
        return measurements[unfairness_metric][0]


class UnfairnessMetric():
    def __init__(self, protected_groups, unfairness_metric):
        """Unfairness metric that can be used with scikit-learn functionality; e.g., in a Pipeline.
        `protected_groups` must be specified in advance so that it can be used later, since
        scikit-learn does not allow for passing in custom values during cross-validation splitting.

        Args:
            protected_groups (pd.Series): Group membership of each person/instance
            unfairness_metric (str): Name of unfairness metric
        """
        assert isinstance(protected_groups, pd.Series), 'pd.Series required for index matching'
        self.protected_groups = protected_groups
        self.unfairness_metric = unfairness_metric
        self.__name__ = 'UnfairnessMetric'

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, pd.Series), 'pd.Series required for index matching'
        test_groups = self.protected_groups.loc[y_true.index]
        return calc_unfairness(y_true, y_pred, test_groups, self.unfairness_metric)


class CombinedMetric():
    def __init__(self, accuracy_metric_func, protected_groups, unfairness_metric,
                 unfairness_weight=1.0):
        """Creates a combined unfairness and accuracy metric that can be used with sciki-learn
        functionality; e.g., in a Pipeline. `protected_groups` must be specified in advance so that
        it can be used later, since scikit-learn does not allow for passing in custom values during
        cross-validation splitting.

        The resulting metric will subtract unfairness from accuracy.

        Args:
            accuracy_metric_func (function): E.g., `sklearn.metrics.accuracy_score`
            protected_groups (pd.Series): Group membership of each person/instance
            unfairness_metric (str): Name of unfairness metric
            unfairness_weight (float): How to weight unfairness in the calculation; [0, inf]
        """
        assert isinstance(protected_groups, pd.Series), 'pd.Series required for index matching'
        self.accuracy_metric_func = accuracy_metric_func
        self.protected_groups = protected_groups
        self.unfairness_metric = unfairness_metric
        self.unfairness_weight = unfairness_weight
        self.__name__ = 'CombinedMetric'

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, pd.Series), 'pd.Series required for index matching'
        test_groups = self.protected_groups.loc[y_true.index]
        unfairness = calc_unfairness(y_true, y_pred, test_groups, self.unfairness_metric)
        return self.accuracy_metric_func(y_true, y_pred) - unfairness * self.unfairness_weight


if __name__ == '__main__':
    # accuracy measure fairness
    v = CombinedMetric(metrics.accuracy_score, pd.Series([0, 1, 1]), 'treatment_equality')
    score = v(pd.Series([0, 1, 1]), pd.Series([0, 0, 1]))
    print(score)

    from sklearn import datasets, naive_bayes, model_selection
    import numpy as np
    np.random.seed(11798)
    from dataset_loader import load_sample_data
    ds = load_sample_data()
    y = pd.Series(ds['labels'])
    protected_groups = pd.Series(ds['data']['group'])
    # X = ds['data'][['fair_feature', 'unfair_feature']]
    X = np.array(ds['data']['unfair_feature']).reshape(-1, 1)

    # X, y = datasets.load_iris(return_X_y=True)
    # y = pd.Series(y)
    # # Since there are systematic biases in per-class accuracy, treating each class as a protected
    # # group lowers the score
    # protected_groups = pd.Series(y)
    # Conversely, randomly generating the protected groups has little effect (as expected)
    # protected_groups = pd.Series(np.random.randint(0, 2, len(y)))
    clf = naive_bayes.GaussianNB()
    cm = CombinedMetric(metrics.accuracy_score, protected_groups, 'all_equality', 1)
    scoring = metrics.make_scorer(cm)
    cross_val = model_selection.KFold(4, shuffle=True)
    scores = model_selection.cross_val_score(clf, X, y, cv=cross_val, scoring=scoring)
    print(scores)
