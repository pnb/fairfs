# Using conda environment "fairfs" (Python 3.8)
import pandas as pd
import numpy as np
from tqdm import tqdm
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import metrics, model_selection, pipeline, preprocessing
from sklearn import linear_model, naive_bayes, tree

import dataset_loader
import unfairness_metrics


PROTECTED_COLUMN = 'group'  # 'group' for simulated data, 'rural' for 2 other datasets
ITERATIONS = 100


def run_experiment(X, y, clf, protected_groups, unfairness_metric, unfairness_weight):
    metric = unfairness_metrics.UnfairnessMetric(protected_groups, unfairness_metric)
    unfairness_scorer = metrics.make_scorer(metric)
    unfairness_means = []
    kappa_means = []
    selected_feature_props = np.zeros([ITERATIONS, X.shape[1]])
    for i in tqdm(range(100), desc=' Training ' + clf.__class__.__name__):
        xval = model_selection.KFold(4, shuffle=True, random_state=i)
        # Make a metric combining accuracy and subtracting unfairness w.r.t. the protected groups
        metric = unfairness_metrics.CombinedMetric(metrics.cohen_kappa_score, protected_groups,
                                                   unfairness_metric, unfairness_weight)
        combined_scorer = metrics.make_scorer(metric)
        sfs = SequentialFeatureSelector(clf, 'best', verbose=0, cv=xval, scoring=combined_scorer,
                                        n_jobs=2)
        pipe = pipeline.Pipeline([
            ('standardize', preprocessing.StandardScaler()),
            ('feature_selection', sfs),
            ('model', clf),
        ])
        result = model_selection.cross_validate(pipe, X, y, verbose=0, cv=xval, scoring={
            'unfairness': unfairness_scorer,
            'kappa': metrics.make_scorer(metrics.cohen_kappa_score),
        }, return_estimator=True)
        unfairness_means.append(result['test_unfairness'].mean())
        kappa_means.append(result['test_kappa'].mean())
        for estimator in result['estimator']:
            for feature_i in estimator.named_steps['feature_selection'].k_feature_idx_:
                selected_feature_props[i][feature_i] += 1 / len(result['estimator'])
    return unfairness_means, kappa_means, selected_feature_props


# ds = dataset_loader.get_uci_student_performance(median_split=True)['uci_student_performance_math']
# print(ds.keys())  # data, labels, participant_ids, feature_names
ds = dataset_loader.get_simulated_data()

# Pick a column to use as the "protected" group labels
protected_col_index = np.nonzero(ds['feature_names'] == PROTECTED_COLUMN)[0][0]
protected_groups = pd.Series(ds['data'][:, protected_col_index])

# Does the method reduce unfairness?
dfs = []
for m in [naive_bayes.GaussianNB(), linear_model.LogisticRegression(random_state=11798),
          tree.DecisionTreeClassifier(random_state=11798)]:
    for unfairness_metric in unfairness_metrics.UNFAIRNESS_METRICS:
        for unfairness_weight in [0, 1, 2, 3, 4]:
            print('Training', m.__class__.__name__)
            print('Unfairness metric:', unfairness_metric)
            print('Unfairness metric weight:', unfairness_weight)
            unfairnesses, kappas, feature_selected_props = run_experiment(
                ds['data'], pd.Series(ds['labels']), m, protected_groups, unfairness_metric,
                unfairness_weight)
            dfs.append(pd.DataFrame({
                'model': [m.__class__.__name__] * len(kappas),
                'unfairness_metric': [unfairness_metric] * len(kappas),
                'unfairness_weight': [unfairness_weight] * len(kappas),
                'iteration': range(1, len(kappas) + 1),
                'unfairness': unfairnesses,
                'kappa': kappas,
                'protected_column_selected_prop': feature_selected_props[:, protected_col_index],
            }))
            # What features does the model favor if it is optimizing for unfairness?
            if 'fair_feature' in ds['feature_names']:  # Synthetic data
                for col in ['fair_feature', 'unfair_feature']:
                    col_index = np.nonzero(ds['feature_names'] == col)[0][0]
                    dfs[-1][col + '_selected_prop'] = feature_selected_props[:, col_index]
            pd.concat(dfs).to_csv('fairfs_results.csv', index=False)
