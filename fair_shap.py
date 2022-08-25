# Using pyenv environment "fairfs" (Python 3.8)
import shap
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from sklearn import metrics, model_selection, pipeline, preprocessing
from sklearn import tree, linear_model, naive_bayes
import unfairness_metrics

from column_threshold_selector import ColumnThresholdSelector

PROTECTED_COLUMN = 'Sex'  # 'Sex' for adult, 'group' for synthetic
PRIVILEGED_VALUE = 1      # 1 for synthetic and for adult (indicates male)
UNPRIVILEGED_VALUE = 0    # 0 for synthetic and for adult (indicates female)
ITERATIONS = 100
ACCURACY_METRIC = metrics.roc_auc_score
# MODEL_LIST = [naive_bayes.GaussianNB(), linear_model.LogisticRegression(random_state=11798),
#               tree.DecisionTreeClassifier(random_state=11798)]
MODEL_LIST = [tree.DecisionTreeClassifier(random_state=11798)]
UNFAIRNESS_METRICS_LIST = unfairness_metrics.UNFAIRNESS_METRICS
SELECTION_METRIC = 'treatment_equality'  # change as desired
SELECTION_CUTOFFS = [.1, .2, .4, .8]


def main():
    dfs = []
    try:
        dfs.append(pd.read_csv('fairfs_results.csv'))
    except FileNotFoundError:
        pass

    X, y = shap.datasets.adult()
    y = pd.Series(y, index=X.index)

    # Pick a column to use as the "protected" group labels
    # protected_col_index = X[PROTECTED_COLUMN]
    protected_groups = X[PROTECTED_COLUMN]

    for m in MODEL_LIST:
        for unfairness_metric in UNFAIRNESS_METRICS_LIST:
            for selection_cutoff in SELECTION_CUTOFFS:
                print('Training', m.__class__.__name__)
                print('Unfairness metric:', unfairness_metric)
                print('Selection cutoff:', selection_cutoff)
                if len(dfs) > 0 and sum((dfs[0].model == m.__class__.__name__)
                                        & (dfs[0].unfairness_metric == unfairness_metric)
                                        & (dfs[0].unfairness_weight == unfairness_weight)) > 0:
                    print('Skipping (already done in output file)')
                    continue

                unfairnesses, aucs, selected_feature_props = run_experiment(X,
                                                                            y,
                                                                            m,
                                                                            protected_groups,
                                                                            PRIVILEGED_VALUE,
                                                                            unfairness_metric,
                                                                            selection_cutoff
                                                                            )

                dfs.append(pd.DataFrame({
                    'model': [m.__class__.__name__] * len(aucs),
                    'unfairness_metric': [unfairness_metric] * len(aucs),
                    'unfairness_weight': [selection_cutoff] * len(aucs),
                    'iteration': range(1, len(aucs) + 1),
                    'unfairness': unfairnesses,
                    'auc': aucs,
                    'protected_column_selected_prop': feature_selected_props[:, protected_col_index],
                }))
                pd.concat(dfs).to_csv('fairfs_results.csv', index=False)


def run_experiment(X, y, model, sensitive_column, privileged_value, unfairness_metric, selection_cutoff):
    # create instance of unfairness metric to pass to scikit result
    metric = unfairness_metrics.UnfairnessMetric(sensitive_column, unfairness_metric)
    # scikit learn function in order to pass as scoring metric in function
    unfairness_scorer = metrics.make_scorer(metric)

    # Create lists to hold fairness and accuracy for each run
    unfairness_means = []
    auc_means = []
    selected_feature_props = np.zeros([ITERATIONS, X.shape[1]])

    for i in tqdm(range(ITERATIONS), desc=' Training ' + model.__class__.__name__):
        # Create 10-fold cross-validation train test split for the overall model
        cross_val = model_selection.KFold(10, shuffle=True, random_state=11798)

        featureSelector = ColumnThresholdSelector(
                model, sensitive_column, privileged_value, selection_cutoff,
                unfairness_metric)

        pipe = pipeline.Pipeline([
            ('feature_selection', featureSelector),
            ('model', model),
        ])

        result = model_selection.cross_validate(pipe, X, y, verbose=0, cv=cross_val, scoring={
            'unfairness': unfairness_scorer,
            'auc': metrics.make_scorer(ACCURACY_METRIC),
        }, return_estimator=True)

        unfairness_means.append(result['test_unfairness'].mean())
        auc_means.append(result['test_auc'].mean())
        for estimator in result['estimator']:
            for feature_i in estimator.named_steps['feature_selection'].k_feature_idx_:
                selected_feature_props[i][feature_i] += 1 / len(result['estimator'])

    # print(result)
    return unfairness_means, auc_means, selected_feature_props


if __name__ == "__main__":
    main()
