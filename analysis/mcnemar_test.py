#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from ast import literal_eval
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib as plt
import scipy
from collections import defaultdict, Counter
from itertools import combinations
import scipy
from statistics import mean, stdev
from math import sqrt
import ast
import os
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="test_tokens_cv/all_tokens/")
args = parser.parse_args()

main_folder = args.root_dir

set_name0 = "only_bert"
set_name1 = "new_prior_confidence_limit_1"
set_name2 = "new_prior_confidence_limit_30"
set_name3 = "new_prior_features_limit_30"

def get_best_token_filename(dirpath):
    files = os.listdir(dirpath)
    files = [os.path.join(dirpath, f) for f in files]
    files.sort(key=lambda x: os.path.getmtime(x))

    for i in range(1, len(files)):
        if files[i].endswith('best_features.json'):
            if files[i-1].endswith('discriminator_training_stats.csv'):
                return files[i-2]
            return files[i-1]
    return None

def get_correct_samples(filename):
    df = pd.read_csv(filename)
    eval_ground_reln = list(df['ground_reln'])
    eval_predicted_reln = list(df['predicted_reln'])
    correct_sample = [int(eval_ground_reln[i] == eval_predicted_reln[i]) for i in range(len(eval_predicted_reln))]
    return correct_sample

def get_samples_from_one_config(set_name0, i):
    subfolder = "surprise_ranker_lr_5e-6_gru_{}_{}".format(set_name0, i)
    dirpath = main_folder + subfolder
    filename = get_best_token_filename(dirpath)
    return get_correct_samples(filename)

def get_all_correct(set_name0):
    all_correct = []
    for j in range(1, 11):
        correct_model = get_samples_from_one_config(set_name0, j)
        all_correct += correct_model
    return all_correct

def get_mcnemar(set_name0, set_name1):
    table = [[0,0], [0,0]]
    ours_over_roberta = []
    for j in range(1, 11):
        correct_model = get_samples_from_one_config(set_name0, j)
        correct_bert = get_samples_from_one_config(set_name1, j)

        for i in range(len(correct_model)):
            our_correct = 1-correct_model[i]
            bert_correct = 1-correct_bert[i]
            if correct_model[i] and not correct_bert[i]:
                ours_over_roberta.append((j,i))
            table[our_correct][bert_correct] += 1

    # calculate mcnemar test
    result = mcnemar(table, exact=True)

    print(table)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

    print('----------')

    return ours_over_roberta


ours_over_roberta = get_mcnemar(set_name1, set_name0)
get_mcnemar(set_name2, set_name0)
get_mcnemar(set_name3, set_name0)
all_correct = get_all_correct(set_name1)
