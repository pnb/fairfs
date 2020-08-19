# Load various datasets and provide them in a unified format for easy experimentation.
# Format of each dataset will be:
# {
#   channel_name: {
#       'data': 2D Numpy array of feature values,
#       'labels': 1D Numpy array of labels (strings),
#       'participant_ids': 1D Numpy array of ids for cross-validation,
#       'feature_names': 1D Numpy array of feature names (strings)
#   }
# }
import pandas as pd
import numpy as np


def get_uci_student_performance(median_split=False):
    # This is a regression dataset on final grade in two different courses.
    math = pd.read_csv('data/uci_student_performance/student-mat.csv', sep=';')
    portuguese = pd.read_csv('data/uci_student_performance/student-por.csv', sep=';')
    for ds in [math, portuguese]:  # Recode some attributes to make nicer features/labels.
        ds['school_id'] = (ds.school.values == 'MS').astype(int)
        ds['male'] = (ds.sex.values == 'M').astype(int)
        ds['rural'] = (ds.address.values == 'R').astype(int)
        ds['famsize_gt3'] = (ds.famsize.values == 'GT3').astype(int)
        ds['parents_cohabitation'] = (ds.Pstatus.values == 'T').astype(int)
        for col in ['Mjob', 'Fjob', 'reason', 'guardian']:
            for v in sorted(ds[col].unique()):
                ds[col + '_' + v] = (ds[col].values == v).astype(int)
            ds.drop(columns=col, inplace=True)
        for col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
                    'romantic']:
            ds[col] = (ds[col].values == 'yes').astype(int)
        ds.drop(columns=['school', 'sex', 'address', 'famsize', 'Pstatus'], inplace=True)
        ds['G3'] = (ds.G3.values - ds.G3.values.min()) / (ds.G3.values.max() - ds.G3.values.min())
        if median_split:
            ds['G3'] = (ds.G3.values > np.median(ds.G3.values)).astype(int)
    return {
        'uci_student_performance_math': {
            'data': math[[f for f in math if f not in ['G1', 'G2', 'G3']]].values,
            'labels': math.G3.values,  # Final grade.
            'participant_ids': np.arange(0, len(math)),  # Every row is one student.
            'feature_names': np.array([f for f in math if f not in ['G1', 'G2', 'G3']])
        },
        'uci_student_performance_portuguese': {
            'data': portuguese[[f for f in portuguese if f not in ['G1', 'G2', 'G3']]].values,
            'labels': portuguese.G3.values,
            'participant_ids': np.arange(0, len(portuguese)),
            'feature_names': np.array([f for f in portuguese if f not in ['G1', 'G2', 'G3']])
        }
    }


def get_all_datasets(median_split_regression=False):
    """Return all datasets, optionally converting regression problems to binary classification.

    Args:
        median_split_regression (bool): Whether or not to convert regression datasets to
            classification datasets by performing a median split.

    Returns:
        dict: mapping of {dataset_name: {'data':, 'labels':, 'participant_ids':, 'feature_names':}}
    """
    return {
        **get_uci_student_performance(median_split_regression),
    }
