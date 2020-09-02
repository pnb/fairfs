# Load various datasets and provide them in a unified format for easy experimentation.
# Format of each dataset will be:
# {
#   dataset_name: {
#       'data': 2D Numpy array of feature values,
#       'labels': 1D Numpy array of labels (strings),
#       'participant_ids': 1D Numpy array of ids for cross-validation,
#       'feature_names': 1D Numpy array of feature names (strings)
#   }
# }
import pandas as pd
import numpy as np
from scipy.io import arff


def get_uci_student_performance(median_split=True):
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


def get_uci_student_academics(median_split=True):
    # This is an ordinal regression dataset with several academic outcome variables:
    #   tnp and twp -- Grade in two exams: Best, Vg (very good), Good, Pass, Fail
    #   iap -- Grade on assessments leading up to the final exam
    #   esp -- End of semester grade
    data, _ = arff.loadarff('data/uci_student_academics/Sapfile1.arff')
    df = pd.DataFrame(data)
    processed = pd.DataFrame({
        'final_grade': df.esp.map({b'Best': 4, b'Vg': 3, b'Good': 2, b'Pass': 1, b'Fail': 0}),
        'male': (df['ge'] == b'M').astype(int),
        # 'caste': df.cst,
        'overdue_papers': (df['arr'] == b'Y').astype(int),
        # 'married': df['ms'] == b'Married',  # Zero variance (all unmarried)
        'rural': (df['ls'] == b'V').astype(int),
        'free_admission': (df['as'] == b'Free').astype(int),
        'family_income': df.fmi.map({b'Vh': 4, b'High': 3, b'Am': 2, b'Medium': 1, b'Low': 0}),
        'family_size': df.fs.map({b'Large': 2, b'Average': 1, b'Small': 0}),
        'father_edu': df.fq.map({b'Il': 0, b'Um': 1, b'10': 2, b'12': 3, b'Degree': 4, b'Pg': 5}),
        'mother_edu': df.mq.map({b'Il': 0, b'Um': 1, b'10': 2, b'12': 3, b'Degree': 4, b'Pg': 5}),
        'father_occupation_service': (df['fo'] == b'Service').astype(int),
        'father_occupation_business': (df['fo'] == b'Business').astype(int),
        'father_occupation_retired': (df['fo'] == b'Retired').astype(int),
        'father_occupation_farmer': (df['fo'] == b'Farmer').astype(int),
        'father_occupation_others': (df['fo'] == b'Others').astype(int),
        'mother_occupation_service': (df['mo'] == b'Service').astype(int),
        'mother_occupation_business': (df['mo'] == b'Business').astype(int),
        'mother_occupation_retired': (df['mo'] == b'Retired').astype(int),
        'mother_occupation_housewife': (df['mo'] == b'Housewife').astype(int),
        'mother_occupation_others': (df['mo'] == b'Others').astype(int),
        'friends': df.nf.map({b'Large': 2, b'Average': 1, b'Small': 0}),
        'study_habits': df.sh.map({b'Good': 2, b'Average': 1, b'Poor': 0}),
        'previous_private_school': (df['ss'] == b'Private').astype(int),
        'instruction_medium_english': (df['me'] == b'Eng').astype(int),
        'instruction_medium_assamese': (df['me'] == b'Asm').astype(int),
        'instruction_medium_hindi': (df['me'] == b'Hin').astype(int),
        'instruction_medium_bengali': (df['me'] == b'Ben').astype(int),
        'travel_time': df.tt.map({b'Large': 2, b'Average': 1, b'Small': 0}),
        'attendance': df.atd.map({b'Good': 2, b'Average': 1, b'Poor': 0}),
    })
    for col in list(processed.columns):
        if processed[col].sum() / len(processed) < .1:
            processed.drop(columns=[col], inplace=True)  # Remove columns with little variance
    if median_split:
        processed.final_grade = (processed.final_grade > processed.final_grade.median()).astype(int)
    return {
        'uci_student_academics': {
            'data': processed[[f for f in processed if f != 'final_grade']].values,
            'labels': processed.final_grade.values,
            'participant_ids': np.arange(0, len(processed)),  # Every row is one student
            'feature_names': np.array([f for f in processed if f != 'final_grade'])
        }
    }


def get_all_datasets(median_split_regression=True):
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


def load_sample_data():
    sample_data = pd.read_csv('data/test_file.csv', header=0)
    X = sample_data.drop('outcome', axis=1)
    y = sample_data['outcome']
    return {'data': X.values,
            'labels': y.values,
            'participant_ids': np.arange(0, len(sample_data)),
            'feature_names': np.array([f for f in sample_data if f not in ['outcome']])
            }


if __name__ == '__main__':
    print(get_uci_student_academics())
