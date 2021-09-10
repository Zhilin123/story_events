#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from ast import literal_eval
import numpy as np
import os
from collections import Counter

import argparse

parser = argparse.ArgumentParser(description='combine features')
parser.add_argument("--mode", default="prev_sent",
                              choices=["prev_sent", "story_cloze"])
args = parser.parse_args()

sentence_considered = args.mode

if sentence_considered == "story_cloze":
    annotation_filename = "cloze_test_val_winter2018_features_combined.csv"
else:
    annotation_filename = "hippocorpus_paragraph_type_and_surprise_annotation_by_sentence_tnlg_sentence.csv"#"hippocorpus_paragraph_type_and_surprise_annotation_by_sentence_with_extracted.csv"

df = pd.read_csv(annotation_filename)

def get_uniques_stable(one_list):
    dict_repr = {i:0 for i in one_list}
    return [i for i in dict_repr]

#utils
def get_fullpath_of_files_in_folder(folder):
    if folder[-1] == "/":
        folder = folder[:-1]

    return [folder + '/' + i for i in os.listdir(folder) if '.DS_Store' not in i]

## get sentences
paragraphs = list(df['paragraph'])
paragraphs = get_uniques_stable(paragraphs)

prior_sentences = []
sentences = []
useful_paragraphs = []

for paragraph in paragraphs:
    sentences += literal_eval(paragraph)[1:]
    prior_sentences += literal_eval(paragraph)[:-1]
    for i in range(len(literal_eval(paragraph)[1:])):
        useful_paragraphs.append(paragraph)

## get annotations
def merge_expected(list_of_terms):
    return ["expected" if "expected" in i else i for i in list_of_terms]

def convert_surprise_annotation_to_numerical(surprise_annotation_per_sentence):
    return [Counter(merge_expected(one_annotation)).most_common(1)[0][0] for one_annotation in surprise_annotation_per_sentence]

def convert_surprise_annotation_uncollated(surprise_annotation_per_sentence):
    return [merge_expected(one_annotation) for one_annotation in surprise_annotation_per_sentence]

surprise_annotations = list(df['surprise_annotations'])

if sentence_considered != "story_cloze":
    surprise_annotations = get_uniques_stable(surprise_annotations)
    surprise_annotation_per_sentence = []

    for surprise_annotation in surprise_annotations:
        paragraph_annotations = literal_eval(surprise_annotation)

        for i in range(1, len(paragraph_annotations[0])):
            sentence_annotation = [paragraph_annotations[j][i] if i < len(paragraph_annotations[j]) else 'noEvent' for j in range(len(paragraph_annotations)) ]
            surprise_annotation_per_sentence.append(sentence_annotation)

    collated_annotation = convert_surprise_annotation_to_numerical(surprise_annotation_per_sentence)
    uncollated_annotation = convert_surprise_annotation_uncollated(surprise_annotation_per_sentence)
else:
    collated_annotation = []
    for i in range(0,len(surprise_annotations), 5):
        collated_annotation += literal_eval(surprise_annotations[i])[1:]

    uncollated_annotation = [[i] for i in collated_annotation]


## get realis events

if sentence_considered != "story_cloze":

    def process_one_csv(filename):
        df = pd.read_csv(filename)
        sentences = list(df['sentence'])
        realis_events = [len(literal_eval(i)) for i in list(df['realis_events'])]

        for i in range(len(sentences)):
            sentence_to_realis_events[sentences[i]] = realis_events[i]

    foldername = "processed_events/"

    sentence_to_realis_events = {}

    realis_filenames = get_fullpath_of_files_in_folder("new_processed_events")

    [process_one_csv(filename) for filename in realis_filenames]

    realis_events = [sentence_to_realis_events[sentence] for sentence in sentences]

else:

    realis_events = list(df['realis'])
    realis_events = [realis_events[i] for i in range(len(realis_events)) if i % 5 > 0]


## sequentuality

if sentence_considered != "story_cloze":

    sentence_to_bag = {}
    sentence_to_chain = {}
    sentence_to_one_prev = {}

    def process_story_level_ppl_and_mem_type_for_annotated(filename):
        df = pd.read_csv(filename)
       
        ppl_bag = list(df['text_xents_hist0'])
        ppl_chain = list(df['text_xents_hist-1'])
        ppl_one_prev = list(df['text_xents_hist1'])
        sents = list(df['sents'])

        for i in range(len(sents)):
            sentence_to_bag[sents[i]] = ppl_bag[i]
            sentence_to_chain[sents[i]] = ppl_chain[i]
            sentence_to_one_prev[sents[i]] = ppl_one_prev[i]


    ppl_filenames = get_fullpath_of_files_in_folder("extracted_annotation_files_gpt2")
    ppl_filenames = [i for i in ppl_filenames if i.endswith(".sentenceLevel.csv") and "w_summary" in i]

    [process_story_level_ppl_and_mem_type_for_annotated(filename) for filename in ppl_filenames]

    ppl_bag = [sentence_to_bag[sentence] if sentence in sentence_to_bag else sentence_to_bag[sentence.split('.')[0]+'.'] for sentence in sentences]
    ppl_chain = [sentence_to_chain[sentence] if sentence in sentence_to_chain else sentence_to_chain[sentence.split('.')[0]+'.'] for sentence in sentences]
    ppl_one_prev = [sentence_to_one_prev[sentence] if sentence in sentence_to_one_prev else sentence_to_one_prev[sentence.split('.')[0]+'.'] for sentence in sentences]

else:
    def filter_first_of_five(some_list):
        return [some_list[i] for i in range(len(some_list)) if i % 5 > 0]

    def process_story_level_ppl_for_annotated(filename):
        df = pd.read_csv(filename)
        ppl_bag = filter_first_of_five(list(df['text_xents_hist0']))
        ppl_chain = filter_first_of_five(list(df['text_xents_hist-1']))
        ppl_one_prev = filter_first_of_five(list(df['text_xents_hist1']))
        sents = filter_first_of_five(list(df['sents']))

        return ppl_bag, ppl_chain, ppl_one_prev, sents


    ppl_filename = "storycloze.gpt2Pplx.-1.0.1.sentenceLevel.csv"
    ppl_bag, ppl_chain, ppl_one_prev, sents = process_story_level_ppl_for_annotated(ppl_filename)


# get confidence of atomic and glucose labels

def get_confidence_from_one_file(filename):
    df = pd.read_csv(filename)
    confidence = list(df['confidence'])
    results = [literal_eval(i) for i in confidence]
    return results

def redistribute(list_of_4_confidences):
    new_confidences = []
    for i in range(len(list_of_4_confidences[0])):
        new_confidence = []
        for j in range(4):
            new_confidence += list_of_4_confidences[j][i][:-1] # removes all xNone
        new_confidences.append(new_confidence)
    return new_confidences

if sentence_considered == "prev_sent":
    folder_name = "pred_with_confidence"
    no_agg_foldername = "no_agg_pool_pred_with_confidence_paragraph"
elif sentence_considered == "story_cloze":
    folder_name = "story_cloze_pred"
    no_agg_foldername = "no_agg_pool_story_cloze_para_pred"


filenames = sorted(get_fullpath_of_files_in_folder(folder_name))
all_confidence = [get_confidence_from_one_file(filename) for filename in filenames]
prior_features_filenames = sorted(get_fullpath_of_files_in_folder(no_agg_foldername))
all_prior_confidence = [get_confidence_from_one_file(filename) for filename in prior_features_filenames]


## get type of event
if sentence_considered != "story_cloze":
    type_of_events = list(df['input_type'])
    prior_sents_in_para = list(df['prior_sentences_in_parapgraph'])
    useful_indices = [i for i in range(len(prior_sents_in_para)) if isinstance(prior_sents_in_para[i],str)]
    memory_type_of_events = [type_of_events[i] for i in useful_indices]


# get cosine similarity to TNLG sentences
if sentence_considered != "story_cloze":
    tnlg_filename = "sentence_cosine_similarity_updated_sbert.csv"
else:
    tnlg_filename = "story_cloze_sentence_cosine_similarity_updated_sbert.csv"

tnlg_df = pd.read_csv(tnlg_filename)
tnlg_similarities = [float(i) for i in list(tnlg_df['cosine_similarities_sentence'])]

all_confidence_linear = [all_confidence[0][i] +
                         all_confidence[1][i] +
                         all_confidence[2][i] +
                         all_confidence[3][i] +
                         [realis_events[i]] +
                         [tnlg_similarities[i]] +
                         [ppl_bag[i]] +
                         [ppl_one_prev[i], (ppl_one_prev[i]-ppl_bag[i])]
                         +[redistribute([all_prior_confidence[j][i] for j in range(4)])]
                         +[uncollated_annotation[i]]
                                for i in range(len(all_confidence[0]))]

atomic_features = ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant", "xNone-atomic"]
reverse_atomic_features = [i+'-r' for i in atomic_features]
glucose_features = ["{}_{}".format(i,"glucoseNL") for i in range(1,11)] + ["xNone-glucose"]
reverse_glucose_features = [i+'-r' for i in glucose_features]
other_features = ["n_realis",
                  "tnlg_cosine_sentence",
                  "ppl_bag",
                  "ppl_one_prev",
                  "ppl_one_prev_minus_bag",
                  "prior_confidence",
                  "uncollated_annotation"]

all_features = atomic_features + reverse_atomic_features + glucose_features + reverse_glucose_features + other_features

def get_story_ids_from_paragraphs(paragraphs):
    story_id = 1
    story_ids = [1]
    for i in range(1, len(paragraphs)):
        if paragraphs[i] != paragraphs[i-1]:
            story_id += 1
        story_ids.append(story_id)
    return story_ids
    
def save_new_df(all_confidence_linear, all_features, sentences, paragraphs, 
                prior_sentences, collated_annotation, save_filename, uncollated_annotation):

    data_dict = {}
    for i in range(len(all_features)):
        data_dict[all_features[i]] = [j[i] for j in all_confidence_linear]
    data_dict['sentence'] = sentences
    data_dict['paragraph'] = paragraphs
    data_dict['prior_sentence'] = prior_sentences
    data_dict['surprise_annotation'] = collated_annotation
    data_dict['story_ids'] = get_story_ids_from_paragraphs(paragraphs)
    new_df = pd.DataFrame.from_dict(data_dict)
    new_df.to_csv(save_filename)

save_new_df(all_confidence_linear, all_features, sentences, useful_paragraphs,
            prior_sentences, collated_annotation,
            "all_features_including_annotations_{}_with_prior_confidence.csv".format(sentence_considered), uncollated_annotation) #
