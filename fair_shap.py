# Using pyenv environment "fairfs" (Python 3.8)
import shap
import pandas as pd
import numpy as np
import math
from sklearn import metrics, model_selection, pipeline, preprocessing
from sklearn import tree
import unfairness_metrics

# import dataset_loader
from column_threshold_selector import ColumnThresholdSelector

PROTECTED_COLUMN = 'Sex'  # 'Sex' for adult, 'group' for synthetic
PRIVILEGED_VALUE = 1      # 1 for synthetic and for adult (indicates male)
UNPRIVILEGED_VALUE = 0    # 0 for synthetic and for adult (indicates female)
ITERATIONS = 100
ACCURACY_METRIC = metrics.roc_auc_score
MODEL_LIST = [naive_bayes.GaussianNB(), linear_model.LogisticRegression(random_state=11798),
              tree.DecisionTreeClassifier(random_state=11798)]
UNFAIRNESS_METRICS_LIST = unfairness_metrics.UNFAIRNESS_METRICS
SELECTION_METRIC = 'treatment_equality'  # change as desired
SELECTION_CUTOFFS = [.1, .2, .3, .4]
CUTOFF_VALUE = 0.2


def main():
    X, y = shap.datasets.adult()
    y = pd.Series(y, index=X.index)

    unfairness_means = []
    auc_means = []

    # model = tree.DecisionTreeClassifier(random_state=11798)

    run_experiment(X, y, model, X[PROTECTED_COLUMN],
                   PRIVILEGED_VALUE, SELECTION_METRIC, CUTOFF_VALUE)


def run_experiment(X, y, model, sensitive_column, privileged_value, unfairness_metric, unfairness_weight):
    # put everything from here to result (line 60) in loop to use different cutoffs and metrics
    # create instance of unfairness metric to pass to scikit result
    metric = unfairness_metrics.UnfairnessMetric(sensitive_column, unfairness_metric)
    # scikit learn function in order to pass as scoring metric in function
    unfairness_scorer = metrics.make_scorer(metric)

    # Create list to hold fairness and accuracy for each run
    # results = []

    # Create 10-fold cross-validation train test split for the overall model
    cross_val = model_selection.KFold(10, shuffle=True, random_state=11798)

    featureSelector = ColumnThresholdSelector(
            model, sensitive_column, privileged_value, unfairness_weight,
            unfairness_metric)

    pipe = pipeline.Pipeline([
        ('feature_selection', featureSelector),
        ('model', model),
    ])

    result = model_selection.cross_validate(pipe, X, y, verbose=0, cv=cross_val, scoring={
        'unfairness': unfairness_scorer,
        'auc': metrics.make_scorer(ACCURACY_METRIC),
    }, return_estimator=True)

    print(result)
    # results.append(pd.Series({
    #         'model': model.__class__.__name__,
    #         'unfairness_metric': SELECTION_METRIC,
    #         'auc': ACCURACY_METRIC(y_test, predictions),
    #         'model_fairness': ColumnThresholdSelector.feature_fairness(X_test, y_test, predictions),
    #         'columns': selected_features,
    #         'cutoff': CUTOFF_VALUE
    #     }))

    # pd.concat(result).to_csv('fairfs_shap_results_column_selector.csv', index=False)


if __name__ == "__main__":
    main()
