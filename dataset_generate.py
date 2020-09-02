import pandas as pd
import numpy as np
import random
from collections import Counter

np.random.seed(10)

index = [0] * 500 + [1] * 500

randomNums = np.random.normal(loc=10, scale=5, size=1000).astype(int)
max_num = max(randomNums)
min_num = min(randomNums)

normalized_outcome = (randomNums - min_num)/(max_num-min_num)
normalized_outcome = np.round(normalized_outcome, 1)
outcome = np.where(normalized_outcome < 0.5, 0, 1)

n_index_fair = range(1000)

fair_index = random.sample(n_index_fair, 300)

fair_feature = []
for i in n_index_fair:
    if i in fair_index:
        fair_feature.append(round(1-normalized_outcome[i], 1))
    else:
        fair_feature.append(normalized_outcome[i])


n_index_unfair_0 = random.sample(range(500), 225)
n_index_unfair_1 = random.sample(range(500, 1000), 75)

unfair_feature = []
for i in range(500):
    if i in n_index_unfair_0:
        unfair_feature.append(round(1-normalized_outcome[i], 1))
    else:
        unfair_feature.append(normalized_outcome[i])
for i in range(500, 1000):
    if i in n_index_unfair_1:
        unfair_feature.append(round(1-normalized_outcome[i], 1))
    else:
        unfair_feature.append(normalized_outcome[i])

gen_file = pd.DataFrame(columns=['group', 'fairness_feature', 'unfairness_feature', 'outcome'])
gen_file['group'] = index
gen_file['fairness_feature'] = fair_feature
gen_file['unfairness_feature'] = unfair_feature
gen_file['outcome'] = outcome

gen_file.to_csv('data/test_file.csv', index=False)

