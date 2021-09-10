#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import scipy
from scipy import stats
from statsmodels.stats import multitest
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filename")
args = parser.parse_args()

filename = args.filename

df = pd.read_csv(filename)
predicted_surprise = list(df["predicted_reln"])
ground_story_cloze = list(df["ground_reln"])

combined_weights = np.zeros((2,3), dtype=int)

for i in range(len(predicted_surprise)):
    combined_weights[int(ground_story_cloze[i]), int(predicted_surprise[i])] += 1

confidence_level = 0.05

pvals = []

def get_p_for_2_distribution_chisq(yes_label, no_label):
    results = np.array([yes_label, no_label])

    chisq, pval, _, _ = scipy.stats.chi2_contingency(results)

    pvals.append(pval)

    return abs(pval) < confidence_level


for col in range(3):
    nonsense_col = [combined_weights[0, col], sum(combined_weights[0])-combined_weights[0, col]]
    commonsense_col = [combined_weights[1, col], sum(combined_weights[1])-combined_weights[1, col]]
    get_p_for_2_distribution_chisq(nonsense_col, commonsense_col)

reject, pvals_corrected, _, _ = multitest.multipletests(pvals, alpha=0.05, method='holm', is_sorted=False, returnsorted=False)
print(reject, pvals_corrected)
