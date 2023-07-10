# Using pyenv environment "fairfs" (Python 3.8)
import shap
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import metrics, model_selection, pipeline
from sklearn import tree, linear_model, naive_bayes
import unfairness_metrics
import dataset_loader
import os

from column_threshold_selector import ColumnThresholdSelector

PROTECTED_COLUMN = 'Sex'  # 'Sex' for adult, 'group' for synthetic
DATASET = 'adult'  # options currently adult or synthetic
FILENAME = 'adult_fairfs_shap_results_07102023.csv'
PRIVILEGED_VALUE = 1      # 1 for synthetic and for adult (indicates male)
UNPRIVILEGED_VALUE = 0    # 0 for synthetic and for adult (indicates female)
ITERATIONS = 100
ACCURACY_METRIC = metrics.roc_auc_score
# MODEL_LIST = [naive_bayes.GaussianNB(), linear_model.LogisticRegression(random_state=11798),
#               tree.DecisionTreeClassifier(random_state=11798)]
MODEL_LIST = [tree.DecisionTreeClassifier(random_state=11798)]
UNFAIRNESS_METRICS_LIST = unfairness_metrics.UNFAIRNESS_METRICS



def main():
    dfs = []
    try:
        dfs.append(pd.read_csv(FILENAME))
    except FileNotFoundError:
        pass

    if DATASET == 'adult':
        print("Using adult dataset")
        X, y_tmp = shap.datasets.adult()
        y = pd.Series(y_tmp, index=X.index)
        SELECTION_CUTOFFS = [.2, .4, .6, .8]

    elif DATASET == 'synthetic':
        print("Using synthetic dataset")
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

                all_results = run_experiment(X,
                                             y,
                                             m,
                                             group_membership,
                                             PRIVILEGED_VALUE,
                                             unfairness_metric,
                                             selection_cutoff
                                             )
                # keep the header the first time we write to the file so we have column names
                if os.path.isfile(FILENAME):
                    all_results.to_csv(FILENAME, mode='a', index=False, header=False)
                else:
                    all_results.to_csv(FILENAME, mode='a', index=False)


def run_experiment(X, y, model, group_membership, privileged_value, unfairness_metric, selection_cutoff):
    # create instance of unfairness metric to pass to scikit result
    metric = unfairness_metrics.UnfairnessMetric(group_membership, unfairness_metric)
    # scikit learn function in order to pass as scoring metric in function
    unfairness_scorer = metrics.make_scorer(metric)

    # Create lists to hold fairness and accuracy for each run
    unfairness_means = []
    auc_means = []
    priv_cms = []
    unpriv_cms = []

    selected_feature_props = pd.DataFrame(data=np.zeros([ITERATIONS, X.shape[1]]),
                                          columns=X.columns
                                          )
    for i in tqdm(range(ITERATIONS), desc=' Training ' + model.__class__.__name__):
        # Create 10-fold cross-validation train test split for the overall model
        cross_val = model_selection.KFold(10, shuffle=True, random_state=i)

        # use i as random seed
        feature_selector = ColumnThresholdSelector(
                model, group_membership, selection_cutoff,
                unfairness_metric, rand_seed=i)

        pipe = pipeline.Pipeline([
            ('feature_selection', feature_selector),
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


        # keep track of averages in confusion matrix across each split
        priv_cm_per_fold = pd.DataFrame(data=np.zeros([10, 4]),
                                        columns=['priv_tn', 'priv_fp', 'priv_fn', 'priv_tp']
                                        )
        unpriv_cm_per_fold = pd.DataFrame(data=np.zeros([10, 4]),
                                          columns=['unpriv_tn', 'unpriv_fp', 'unpriv_fn', 'unpriv_tp']
                                          )
        i = 0
        for fold_i, (train_i, test_i) in enumerate(cross_val.split(X, y)):
            estimator = result['estimator'][fold_i]
            # reset the index so these are within the range of the split and can be used with the predictions
            test_x = X.iloc[test_i].reset_index(drop=True)
            test_y = y.iloc[test_i].reset_index(drop=True)
            priv_index = test_x[test_x[PROTECTED_COLUMN] == privileged_value].index
            unpriv_index = test_x[test_x[PROTECTED_COLUMN] != privileged_value].index
            predictions = estimator.predict(test_x)

            # get confusion matrix for each group
            matrix_priv = metrics.confusion_matrix(test_y.iloc[priv_index], predictions[priv_index])  # subset for priv
            matrix_unpriv = metrics.confusion_matrix(test_y[unpriv_index], predictions[unpriv_index])  # subset for unrpiv

            priv_cm_per_fold.iloc[i] = matrix_priv.reshape(1, 4)
            unpriv_cm_per_fold.iloc[i] = matrix_unpriv.reshape(1, 4)
            i += 1

        priv_means = priv_cm_per_fold.mean()
        priv_cms.append(priv_means.values)
        unpriv_means = unpriv_cm_per_fold.mean()
        unpriv_cms.append(unpriv_means.values)

    priv_df = pd.DataFrame(priv_cms, columns=['priv_tn', 'priv_fp', 'priv_fn', 'priv_tp'])
    unpriv_df = pd.DataFrame(unpriv_cms, columns=['unpriv_tn', 'unpriv_fp', 'unpriv_fn', 'unpriv_tp'])

    results_df = pd.DataFrame({
                    'model': [model.__class__.__name__] * len(auc_means),
                    'unfairness_metric': [unfairness_metric] * len(auc_means),
                    'cutoff_value': [selection_cutoff] * len(auc_means),
                    'iteration': range(1, len(auc_means) + 1),
                    'unfairness': unfairness_means,
                    'auc': auc_means,
                })

    all_results = pd.concat([results_df, priv_df, unpriv_df, selected_feature_props], axis='columns')
    return all_results


if __name__ == "__main__":
    main()
