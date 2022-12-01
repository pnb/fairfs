# Using pyenv environment "fairfs" (Python 3.8)
import shap
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import metrics, model_selection, pipeline
from sklearn import tree, linear_model, naive_bayes
import unfairness_metrics
import dataset_loader

from column_threshold_selector import ColumnThresholdSelector

PROTECTED_COLUMN = 'group'  # 'Sex' for adult, 'group' for synthetic
DATASET = 'synthetic'  # options currently adult or synthetic
FILENAME_STR = 'fairfs_shap_results_11302022_synthetic'
FILENAME = FILENAME_STR + '.csv'
FEATURE_FILENAME = FILENAME_STR + '_selected_features.csv'
PRIVILEGED_VALUE = 1      # 1 for synthetic and for adult (indicates male)
UNPRIVILEGED_VALUE = 0    # 0 for synthetic and for adult (indicates female)
ITERATIONS = 100
ACCURACY_METRIC = metrics.roc_auc_score
# MODEL_LIST = [naive_bayes.GaussianNB(), linear_model.LogisticRegression(random_state=11798),
#               tree.DecisionTreeClassifier(random_state=11798)]
MODEL_LIST = [tree.DecisionTreeClassifier(random_state=11798)]
UNFAIRNESS_METRICS_LIST = unfairness_metrics.UNFAIRNESS_METRICS
SELECTION_CUTOFFS = [.1, .2, .4, .8]


def main():
    dfs = []
    feature_selection_dfs = []
    try:
        dfs.append(pd.read_csv(FILENAME))
    except FileNotFoundError:
        pass

    try:
        feature_selection_dfs.append(pd.read_csv(FEATURE_FILENAME))
    except FileNotFoundError:
        pass

    if DATASET == 'adult':
        X, y_tmp = shap.datasets.adult()
        y = pd.Series(y_tmp, index=X.index)

    if DATASET == 'synthetic':
        ds = dataset_loader.get_simulated_data()['simulated_data']
        X = pd.DataFrame(ds['data'], columns=ds['feature_names'])
        y = pd.Series(ds['labels'])
        SELECTION_CUTOFFS = [.4, .8]  # only 3 columns, so values smaller than .4 will select no features

    else:
        print("Please select which dataset you are using")

    # Pick the column(s) of interest to use as the group labels
    group_membership = X[PROTECTED_COLUMN]

    for m in MODEL_LIST:
        for unfairness_metric in UNFAIRNESS_METRICS_LIST:
            for selection_cutoff in SELECTION_CUTOFFS:
                print('Training', m.__class__.__name__)
                print('Unfairness metric:', unfairness_metric)
                print('Selection cutoff:', selection_cutoff)
                if len(dfs) > 0 and sum((dfs[0].model == m.__class__.__name__)
                                        & (dfs[0].unfairness_metric == unfairness_metric)
                                        & (dfs[0].cutoff_value == selection_cutoff)) > 0:
                    print('Skipping (already done in output file)')
                    continue

                unfairnesses, aucs, selected_feature_props = run_experiment(X,
                                                                            y,
                                                                            m,
                                                                            group_membership,
                                                                            PRIVILEGED_VALUE,
                                                                            unfairness_metric,
                                                                            selection_cutoff
                                                                            )

                dfs.append(pd.DataFrame({
                    'model': [m.__class__.__name__] * len(aucs),
                    'unfairness_metric': [unfairness_metric] * len(aucs),
                    'cutoff_value': [selection_cutoff] * len(aucs),
                    'iteration': range(1, len(aucs) + 1),
                    'unfairness': unfairnesses,
                    'auc': aucs,
                }))
                pd.concat(dfs).to_csv(FILENAME, index=False)
                feature_selection_dfs.append(selected_feature_props)
                pd.concat(feature_selection_dfs).to_csv(FEATURE_FILENAME, index=False)


def run_experiment(X, y, model, group_membership, privileged_value, unfairness_metric, selection_cutoff):
    # create instance of unfairness metric to pass to scikit result
    metric = unfairness_metrics.UnfairnessMetric(group_membership, unfairness_metric)
    # scikit learn function in order to pass as scoring metric in function
    unfairness_scorer = metrics.make_scorer(metric)

    # Create lists to hold fairness and accuracy for each run
    unfairness_means = []
    auc_means = []
    selected_feature_props = pd.DataFrame(data=np.zeros([ITERATIONS, X.shape[1]]),
                                          columns=X.columns
                                          )
    for i in tqdm(range(ITERATIONS), desc=' Training ' + model.__class__.__name__):
        # Create 10-fold cross-validation train test split for the overall model
        # TODO: see if doing 4-fold helps with speed up
        cross_val = model_selection.KFold(10, shuffle=True, random_state=i)

        # use i as random seed
        featureSelector = ColumnThresholdSelector(
                model, group_membership, privileged_value, selection_cutoff,
                unfairness_metric, rand_seed=i)

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
            for feature_i in estimator.named_steps['feature_selection'].selected_features:
                selected_feature_props[feature_i][i] += 1 / len(result['estimator'])

    return unfairness_means, auc_means, selected_feature_props


if __name__ == "__main__":
    main()
