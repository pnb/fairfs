import pandas as pd
from scipy.stats import pearsonr
import scipy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
import random

# ensure plots are saved with correct font type
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

DATASETS = ['fairfs_shap_results_11112022.csv']

# Titles in same order as datasets
TITLES = [
    # 'Simulated Data',
    'Adult'  # ,
    # 'Student Academics',
    # 'Student Performance (Math)',
    # 'Student Performance (Portuguese)',
    # 'Simulated Data - Reweighed'

]

DATA_DICT = {}
# create dictionary of titles and datasets to use for graph titles
for i in range(len(DATASETS)):
    # DATA_DICT[TITLES[i]] = 'results/' + DATASETS[i]
    DATA_DICT[TITLES[i]] = DATASETS[i]


# read in data
def create_data_arrays(sheet_name):
    results = pd.read_csv(sheet_name, sep=',')
    return results


# create a plot with each unfairness metric.
# Note this is hardcoded in the order they run in the experiment
def create_plot(title, results, location):
    x = ["10%", "20%", "40%", "80%"]
    title = title
    plt.title(title)
    plt.xlabel("Percentage of Features Kept")
    plt.ylabel("Result (Mean Over 100 Trials)")

    # overall accuracy
    plt.plot(x, results[0:4], ':o', color="orange", label="Overall Accuracy")
    overall = Line2D([0], [0], color='orange', linestyle=":", label="Overall Accuracy")

    # statistical parity
    plt.plot(x, results[4:8], '-o', color="black", label="Statistical Parity")
    statistical = Line2D([0], [0], color='black', linestyle="-", label="Statistical Parity")

    # conditional procedure
    plt.plot(x, results[8:12], '--o', color='red', label="Conditional Procedure")
    condition = Line2D([0], [0], color='red', linestyle="--", label="Conditional Procedure")

    # conditional use accuracy equality
    plt.plot(x, results[12:16], '-.o', color='green', label="Conditional Use Accuracy")
    use_accuracy = Line2D([0], [0], color='green', linestyle="-.", label="Conditional Use Accuracy")

    # treatment equality
    plt.plot(x, results[16:20], '--o', color="cyan", label="Treatment Equality")
    treatment = Line2D([0], [0], color='cyan', linestyle="--", label="Treatment Equality")

    # all equality
    plt.plot(x, results[20:24], '-o', color='blue', label="Total Average Equality")
    total = Line2D([0], [0], color='blue', linestyle="-", label="Total Average Equality")

    plt.legend(handles=[total, condition, use_accuracy, overall,
               statistical, treatment], loc=location, prop={'size': 8})

    plt.savefig(title + '.pdf', dpi=200)
    plt.clf()


def make_all_unfairness_plots():
    for key, value in DATA_DICT.items():
        results = create_data_arrays(value)
        # calculate mean unfairness using unfairness column
        means = calc_mean(results, 'unfairness')
        # create_plot("Gaussian NB - " + key + " - Unfairness", means[0:30], 'upper right')
        # create_plot("Logistic Regression - " + key + " - Unfairness", means[30:60], 'upper right')
        create_plot("Decision Tree - " + key + " - Unfairness_test", means[0:24], 'upper right')


def make_all_accuracy_plots():
    for key, value in DATA_DICT.items():
        results = create_data_arrays(value)
        # calculate mean accuracy using AUC column
        means = calc_mean(results, 'auc')
        # create_plot("Gaussian NB - " + key + " - Accuracy", means[0:30], 'lower left')
        # create_plot("Logistic Regression - " + key + " - Accuracy", means[30:60], 'lower left')
        create_plot("Decision Tree - " + key + " - Accuracy_test", means[0:24], 'lower left')


def calc_mean(results, col):
    means = []
    # hard coded for current number of experiments
    # 100 iterations per unfairness weight + method and 9000 total rows per dataset
    # change if needed
    for i in range(0, 2400, 100):
        data = results[col][i:i + 99]
        mean = np.mean(data)
        means.append(mean)
    return means


# def all_correlation():
#     correlations = {}
#     for key, value in DATA_DICT.items():
#         results = create_data_arrays(value)
#         correlations[key] = calc_correl(results)
#     return correlations


# def calc_correl(results):
#     # hard coded for current number of experiments
#     # 500 total iterations per method and 9000 total columns per dataset
#     # change if needed
#     correlations = []
#     for i in range(0, 9000, 500):
#         auc = results['auc'][i:i + 499]
#         unfairness = results['unfairness'][i:i + 499]
#         corr, p = pearsonr(auc, unfairness)
#         correlations.append({
#             'model': results['model'][i],
#             'metric': results['unfairness_metric'][i],
#             'correlation': corr,
#             'p-value': p
#         })
#     return correlations


# def plot_correlations():
#     results = create_data_arrays(DATA_DICT['Student Academics'])
#     # scatter plot a few correlations as an example.
#     # randomly select indices to plot (can hardcode if desired)
#     # indices = [2000, 5500, 7000] # plots used for AIES paper
#     indices = random.sample(range(0, 9000, 500), 3)
#     for i in indices:
#         auc = results['auc'][i:i + 499]
#         unfairness = results['unfairness'][i:i + 499]
#         plt.scatter(auc, unfairness, color='gray')
#         # fit linear regression line
#         m, b = np.polyfit(auc, unfairness, 1)
#         plt.plot(auc, m * auc + b, color='black')
#         model = results['model'][i]
#         metric = results['unfairness_metric'][i]
#         plt.xlabel("Accuracy")
#         plt.ylabel("Unfairness")
#         plt.title('Student Academics Correlation')
#         plt.savefig(str(i) + model + '_' + metric + '_student_academics.pdf', dpi=200)
#         plt.clf()


if __name__ == '__main__':
    make_all_accuracy_plots()
    make_all_unfairness_plots()
    # all_correlation()
    # plot_correlations()
