#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, random_split
from tqdm import trange
import torch
from torch import nn
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from ast import literal_eval
import random
import collections
from collections import Counter
import copy


class SurpriseDataset(Dataset):

    def __init__(self, filename,  tokenizer, max_length=128, debug_mode=False,
                 data_subset="trn", paragraph_level=False, prior_features_length_limit=30,
                 feature_mode="features", story_cloze=False, no_filter=False, kl_div=False,
                 remove_glucose_reverse=False, remove_tnlg=False, k_fold=False,
                 fold_number=1, total_folds=5, remove_equal=False):

        assert data_subset in ["trn", "tst", "dev", "all"]

        self.kl_div = kl_div
        self.tokenizer = tokenizer
        self.max_length = max_length if not paragraph_level else 512
        self.remove_glucose_reverse = remove_glucose_reverse
        self.k_fold = k_fold
        self.fold_number = fold_number
        self.total_folds = total_folds
        self.remove_equal = remove_equal

        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.features = []
        self.sep_positions = []

        self.data_subset = data_subset

        df = pd.read_csv(filename)

        self.data_category = self.get_data_category_splits(df)
        self.element_to_tokenized = {}

        sentences = list(df['sentence'])
        prior_sentences = list(df['prior_sentence'])
        labels = list(df['surprise_annotation'])
        if not story_cloze:
            uncollated_labels = [literal_eval(i) for i in list(df['uncollated_annotation'])]
        prior_confidence = [literal_eval(i) for i in list(df['prior_confidence'])]
        paragraphs = [literal_eval(i) for i in list(df['paragraph'])]
        prior_labels = self.get_prior_in_paragraph(labels, paragraphs)

        self.labels_set = ['noEvent', 'expected', 'major-surprising']

        non_feature_names = ['Unnamed: 0','sentence', 'paragraph', 'prior_sentence',
                             'surprise_annotation', 'prior_confidence', "story_id",
                             "memory_type", "uncollated_annotation",
                             "xNone-atomic", "xNone-atomic-r",
                             "xNone-glucose", "xNone-glucose-r"]

        if remove_tnlg:
            non_feature_names.append("tnlg_cosine_sentence")
        if remove_glucose_reverse:
            non_feature_names += ["{}_glucoseNL-r".format(i) for i in range(1, 11)]

        feature_names = [i for i in list(df.columns) if i not in non_feature_names]

        print(feature_names)

        features = [list(df[feature_name]) for feature_name in feature_names]
        # reshape features
        features = [[features[j][i] for j in range(len(features))] for i in range(len(features[0]))]
        prior_features = self.get_prior_in_paragraph(features, paragraphs)

        ## filter out first three out of four

        if story_cloze:
            self.labels_set = [0, 1]
            if no_filter:
                labels = [i if i<2 else 1 for i in labels]
            else:
                sentences = self.filter_out_three_sentences(sentences)
                prior_sentences = self.filter_out_three_sentences(prior_sentences)
                labels = self.filter_out_three_sentences(labels)
                paragraphs = self.filter_out_three_sentences(paragraphs)
                prior_labels = self.filter_out_three_sentences(prior_labels)
                features = self.filter_out_three_sentences(features)
                prior_features = self.filter_out_three_sentences(prior_features)
                prior_confidence = self.filter_out_three_sentences(prior_confidence)

        else:
            self.memory_type = self.filter_using_data_subset(list(df["memory_type"]))
            self.uncollated_annotation = self.filter_using_data_subset([literal_eval(i) for i in list(df["uncollated_annotation"])])


        sentences = self.filter_using_data_subset(sentences)
        prior_sentences = self.filter_using_data_subset(prior_sentences)
        labels = self.filter_using_data_subset(labels)
        if not story_cloze:
            uncollated_labels = self.filter_using_data_subset(uncollated_labels)
        paragraphs = self.filter_using_data_subset(paragraphs)
        features = self.filter_using_data_subset(features)
        self.prior_confidence = self.filter_using_data_subset(prior_confidence)
        self.prior_labels = self.filter_using_data_subset(prior_labels)
        self.prior_features = self.filter_using_data_subset(prior_features)


        features = torch.FloatTensor(features)

        self.prior_confidence = [i[-prior_features_length_limit:] for i in self.prior_confidence]
        self.prior_features = [i[-prior_features_length_limit:] for i in self.prior_features]

        print("confidence len: ", len(self.prior_confidence[0][0]))
        self.prior_confidence_and_features = [self.join_prior_confidence_and_features(
                                                self.prior_confidence[i],
                                                self.prior_features[i]) for i in range(len(self.prior_confidence))]

        self.prior_confidence_and_features = [torch.FloatTensor(i) for i in self.prior_confidence_and_features]

        self.prior_confidence = [torch.FloatTensor(i) for i in self.prior_confidence]
        self.prior_features = [torch.FloatTensor(i) for i in self.prior_features]

        self.label_to_key = {self.labels_set[i]:i for i in range(len(self.labels_set))}

        if debug_mode:
            sentences = sentences[:100]

        if feature_mode == "features":
            feature_col = features

        elif feature_mode == "prior_features":
            feature_col = self.prior_features
        else: # feature_mode prior_confidence
            feature_col = self.prior_confidence_and_features

        for i in trange(len(sentences)):
            prior_sentence = self.get_all_prior_sentences_in_para(paragraphs[i], sentences[i]) if paragraph_level else prior_sentences[i]
            if self.kl_div:
                self.add_event(prior_sentence, sentences[i], uncollated_labels[i], feature_col[i])
            else:
                self.add_event(prior_sentence, sentences[i], labels[i], feature_col[i])

        del self.element_to_tokenized

        if not self.kl_div:
            self.loss_weights = self.get_loss_weights()

        self.sentences = sentences

    def filter_out_three_sentences(self, X):
        return [X[i] for i in range(len(X)) if i % 4 == 3]

    def join_prior_confidence_and_features(self, confidence, feature):
        if self.remove_glucose_reverse:
            return [confidence[i][:-10] + feature[i][len(confidence[i][:-10]):] for i in range(len(confidence))]
        return [confidence[i] + feature[i][len(confidence[i]):] for i in range(len(confidence))]


    def get_prior_in_paragraph(self, features, paragraphs):

        features_in_all_paragraphs = []
        features_in_one_paragraph = [features[0]]
        features_in_all_paragraphs.append(copy.copy(features_in_one_paragraph))
        for i in range(1, len(paragraphs)):
            if paragraphs[i-1] == paragraphs[i]:
                features_in_one_paragraph.append(features[i])
            else:
                features_in_one_paragraph = [features[i]]
            features_in_all_paragraphs.append(copy.copy(features_in_one_paragraph))
        return features_in_all_paragraphs


    def get_loss_weights(self):
        label_frequency = Counter(self.labels)
        freq_table = [0 for i in range(len(self.labels_set))]
        for i in label_frequency:
            freq_table[i] = label_frequency[i]
        proportion = [i/sum(freq_table) for i in freq_table]
        loss_weight = [1/i for i in proportion]
        # ensure that all losses add up to 1 on average
        normalized_loss_weight = [i*len(self.labels_set)/sum(loss_weight) for i in loss_weight]
        return torch.FloatTensor(normalized_loss_weight)


    def get_all_prior_sentences_in_para(self, paragraph, sentence):
        sentence_index = [i for i in range(len(paragraph)) if sentence == paragraph[i]][-1]
        ten_sentences_before = ' '.join(paragraph[max(0,sentence_index-10):sentence_index])
        return ten_sentences_before


    def get_data_category_splits_kfold(self, df):
        one_unit = len(df) // self.total_folds
        split_lengths = [one_unit] * self.total_folds
        split_lengths[0] += len(df)%self.total_folds

        all_folds = random_split(range(len(df)),
                                 split_lengths,
                                 generator=torch.Generator().manual_seed(42))

        dev_indices = all_folds[self.fold_number-1]
        data_category = ["trn"] * len(df)
        for dev_index in dev_indices:
            data_category[dev_index] = "dev"
        return data_category


    def get_data_category_splits(self, df):
        if self.k_fold:
            return self.get_data_category_splits_kfold(df)

        one_unit = len(df) // 10
        trn_indices, tst_indices, dev_indices = random_split(range(len(df)),
                                                             [one_unit*8+len(df)%10,one_unit,one_unit],
                                                             generator=torch.Generator().manual_seed(42))

        data_category = [0] * len(df)

        for trn_index in trn_indices:
            data_category[trn_index] = "trn"
        for tst_index in tst_indices:
            data_category[tst_index] = "tst"
        for dev_index in dev_indices:
            data_category[dev_index] = "dev"
        return data_category

    def add_event(self, head_node, tail_node, allowed_label, feature):

        ground_elements = [head_node,"[SEP]",tail_node]


        if tuple(ground_elements) not in self.element_to_tokenized:
            ground_dict = self.tokenizer(" ".join(ground_elements),
                                            truncation=True,
                                            max_length=self.max_length,
                                            padding="max_length",
                                            return_tensors="pt")

            self.element_to_tokenized[tuple(ground_elements)] = ground_dict
        else:
            ground_dict = self.element_to_tokenized[tuple(ground_elements)]

        ground_input_ids = torch.squeeze(ground_dict['input_ids'])
        ground_attn_masks = torch.squeeze(ground_dict['attention_mask'])

        sep_positions = torch.IntTensor([i for i in range(len(ground_input_ids)) if ground_input_ids[i] == self.tokenizer.sep_token_id][1:])

        if not self.kl_div:
            label = self.label_to_key[allowed_label]
        else:
            label = self.get_proportion_of_each(allowed_label)

        if self.kl_div and self.remove_equal and sorted(label)[-1] == sorted(label)[-2]:
            return


        self.labels.append(label)
        self.input_ids.append(ground_input_ids)
        self.attn_masks.append(ground_attn_masks)
        self.features.append(feature)
        self.sep_positions.append(sep_positions)


    def get_proportion_of_each(self, list_of_annot):
        proportion = [0 ,0 ,0]
        for annot in list_of_annot:
            if annot == "noEvent":
                proportion[0] += 1
            elif annot == 'expected':
                proportion[1] += 1
            elif annot == 'major-surprising':
                proportion[2] += 1

        return [i/len(list_of_annot) for i in proportion]

    def filter_using_data_subset(self, original_data):
        if self.data_subset == "all":
            return original_data
        return [original_data[i] for i in range(len(original_data)) if self.data_category[i] == self.data_subset]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.attn_masks[item], self.labels[item], self.features[item], self.sep_positions[item]

    def get_uniques_stable(self, one_list):
        dict_repr = {i:0 for i in one_list}
        return [i for i in dict_repr]

def join_list_of_list(list_of_list):
    new_list = []
    for one_list in list_of_list:
        new_list += one_list
    return new_list

def get_avg_words_incl_tied(sentences, labels):
    # avg words
    total_words = [[] for i in range(4)]
    # proportion
    total_proportion = [[] for i in range(4)]

    for i in range(len(labels)):
        max_i = np.argmax(labels[i]) if isinstance(labels[i], list) else labels[i]

        if isinstance(labels[i], list):
            sorted_labels = sorted(labels[i])
            if sorted_labels[-1] == sorted_labels[-2]:
                max_i = 3
            else:
                max_i = np.argmax(labels[i])

        #samples[max_i] += 1
        total_words[max_i].append(len(sentences[i].split()))
        total_proportion[max_i].append(labels[i][np.argmax(labels[i])] if isinstance(labels[i], list) else labels[i])

    # get total
    #samples = [sum(samples)] + samples
    total_words = [join_list_of_list(total_words)]+ total_words
    total_proportion = [join_list_of_list(total_proportion)] + total_proportion

    for i in range(len(total_words)):
        samples = len(total_words[i])
        proportion = round(samples/len(total_words[0])*100,1)
        avg_words = round(np.mean(total_words[i]),1)
        words_std = round(np.std(total_words[i]),1)
        avg_proportion = round(np.mean(total_proportion[i]) * 100, 1)
        proportion_std = round(np.std(total_proportion[i])* 100, 1)
        print("{} ({}) & {} ({}) & {} ({})".format(samples, proportion, avg_words, words_std, avg_proportion, proportion_std))

if __name__ == "__main__":
    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    tokenizer.add_special_tokens({
                              'pad_token': '[PAD]',
                              'bos_token': '[CLS]',
                              'eos_token': '[PAD]',
                              'sep_token': '[SEP]'
                              })

    filename = "../all_features_including_annotations_prev_sent_with_prior_confidence.csv"

    all_labels = []
    all_sentences = []


    for fold_number in range(1,11):
        dataset = SurpriseDataset(
                    filename,  tokenizer, max_length=128, debug_mode=False,
                    data_subset="dev",
                    no_filter=True,kl_div=True,remove_equal=False,
                    k_fold=True, fold_number=fold_number, total_folds=10
                    #story_cloze=True,

                )

        all_labels += dataset.labels
        all_sentences += dataset.sentences
        
    get_avg_words_incl_tied(all_sentences, all_labels)
