# Using conda environment "fairfs" (Python 3.8)
import pandas as pd
import numpy as np
from tqdm import tqdm
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import metrics, model_selection, pipeline, preprocessing
from sklearn import linear_model, naive_bayes, ensemble

import dataset_loader
import unfairness_metrics


def run_experiment(X, y, clf, protected_groups, unfairness_metric, unfairness_weight):
    metric = unfairness_metrics.UnfairnessMetric(protected_groups, unfairness_metric)
    unfairness_scorer = metrics.make_scorer(metric)
    unfairness_means = []
    kappa_means = []
    for i in tqdm(range(1000), desc=' Training ' + clf.__class__.__name__):
        xval = model_selection.KFold(4, shuffle=True, random_state=i)
        # Make a metric combining accuracy and subtracting unfairness w.r.t. the protected groups
        metric = unfairness_metrics.CombinedMetric(metrics.cohen_kappa_score, protected_groups,
                                                   unfairness_metric, unfairness_weight)
        combined_scorer = metrics.make_scorer(metric)
        sfs = SequentialFeatureSelector(clf, 'best', verbose=0, cv=xval, scoring=combined_scorer)
        pipe = pipeline.Pipeline([
            ('standardize', preprocessing.StandardScaler()),
            ('feature_selection', sfs),
            ('model', clf),
        ])
        result = model_selection.cross_validate(pipe, X, y, verbose=0, cv=xval, scoring={
            'unfairness': unfairness_scorer,
            'kappa': metrics.make_scorer(metrics.cohen_kappa_score),
        })
        unfairness_means.append(result['test_unfairness'].mean())
        kappa_means.append(result['test_kappa'].mean())
    return unfairness_means, kappa_means


ds = dataset_loader.get_uci_student_performance(median_split=True)['uci_student_performance_math']
# print(ds.keys())  # data, labels, participant_ids, feature_names

# Pick a column to use as the "protected" group labels
protected_groups = pd.Series(ds['data'][:, ds['feature_names'] == 'rural'].T[0])

# RQ1: Does the method reduce unfairness?
dfs = []
for m in [naive_bayes.GaussianNB(), linear_model.LogisticRegression(),
          ensemble.RandomForestClassifier()]:
    for unfairness_metric in unfairness_metrics.UNFAIRNESS_METRICS:
        for unfairness_weight in [0, .5, 1, 2, 4]:
            print('Training', m.__class__.__name__)
            print('Unfairness metric:', unfairness_metric)
            print('Unfairness metric weight:', unfairness_weight)
            unfairnesses, kappas = run_experiment(ds['data'], pd.Series(ds['labels']), m,
                                                  protected_groups, unfairness_metric,
                                                  unfairness_weight)
            dfs.append(pd.DataFrame({
                'model': [m.__class__.__name__] * len(kappas),
                'unfairness_metric': [unfairness_metric] * len(kappas),
                'unfairness_weight': [unfairness_weight] * len(kappas),
                'iteration': range(1, len(kappas) + 1),
                'unfairness': unfairnesses,
                'kappa': kappas
            }))
            pd.concat(dfs).to_csv('fairfs_results.csv', index=False)

# RQ2: What features does the model favor if it is optimizing for unfairness?

# Select features using the combined metric
# sfs.fit(X, y)
# print()
# print(len(sfs.k_feature_idx_), 'of', len(ds['feature_names']), 'features selected')
# selected_features = [ds['feature_names'][i] for i in sfs.k_feature_idx_]
# print(selected_features)
