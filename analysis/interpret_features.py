#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="test_tokens_cv/all_tokens/")
args = parser.parse_args()

main_folder = args.root_dir

feature_names = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect',
                 'xIntent', 'xNeed', 'xReact', 'xWant', 'oEffect-r',
                 'oReact-r', 'oWant-r', 'xAttr-r', 'xEffect-r',
                 'xIntent-r', 'xNeed-r', 'xReact-r', 'xWant-r']+ \
                 ["Dim-{}".format(i) for i in range(1,11)] + \
                 ['Realis', 'SimGen',
                 'NLL_bag', 'NLL_prev', 'Sequentiality']

groups = ['atomic']*18 + ['glucose']*10 + ['realis'] + ['predict'] + ['sequential']*3

set_name1 = "new_prior_confidence_limit_1"

def get_samples_from_one_config(set_name, i):
    subfolder = "surprise_ranker_lr_5e-6_gru_{}_{}".format(set_name, i)
    dirpath = main_folder + subfolder
    filename = get_best_token_filename(dirpath)
    return filename

def get_best_token_filename(dirpath):
    files = os.listdir(dirpath)
    files = [os.path.join(dirpath, f) for f in files]
    files.sort(key=lambda x: os.path.getmtime(x))
    for i in range(1, len(files)):
        if files[i].endswith('best_features.json'):
            return files[i]
    return None

def read_file(filename):
    with open(filename) as f:
      data = json.load(f)
    # this is a 3 by 33 list
    return np.array(data['classifier'])

def sort_features_and_feature_names(all_features, feature_names, groups):
    names_and_features = []
    features_list = all_features.tolist()
    for i in range(len(features_list)):
        names_and_features.append([feature_names[i], groups[i]] + features_list[i])
    names_and_features.sort(key=lambda x: (x[1],-x[-1]),reverse=False)
    feature_names_new = []
    features_list_new = []
    for i in range(len(names_and_features)):
        feature_names_new.append(names_and_features[i][0])
        features_list_new.append(names_and_features[i][2:])
    return np.array(features_list_new), feature_names_new

classifier_weights = []
for i in range(1,11):
    filename = get_samples_from_one_config(set_name1, i)
    one_weights = read_file(filename)
    classifier_weights.append(one_weights)

combined_weights = np.mean(np.array(classifier_weights), axis=0)
all_features = combined_weights.T

all_features, feature_names = sort_features_and_feature_names(all_features, feature_names, groups)

all_features = np.array(all_features)

x_axis_names = ["no event", "expected", "surprising"]

plt.subplots(figsize=(10,10))

ax = sns.heatmap(all_features, yticklabels=feature_names, cmap="coolwarm")

ax.set_xticklabels(x_axis_names, rotation=0)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.show()
