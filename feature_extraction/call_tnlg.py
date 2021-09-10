#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd
from tqdm import tqdm
import time
import argparse

parser = argparse.ArgumentParser(description='call tnlg')
parser.add_argument("--mode", default="surprise",
                              choices=["surprise", "story_cloze"])
parser.add_argument("--tnlg_access_url_and_access_key")

args = parser.parse_args()

mode = args.mode
url = args.tnlg_access_url_and_access_key

input_to_response = {}
def continue_one_sentence(sentence):
    if sentence in input_to_response:
        return input_to_response[sentence]
    
    payload = {
        'prompt': sentence,
        'return_logits': False,  
        'return_token_logprobs': False, 
        'top_p': 0.85,
        'temperature': 1.0,
        'response_length': 64
    }
    response = requests.request("POST", url, data=payload)
    input_to_response[sentence] = response.json()['result']
    return input_to_response[sentence]


def get_prior_sentences_in_paragraph(paragraphs, sentences):
    prior_sentences_in_paragraph = []
    paragraph_sentences = []
    for i in range(len(paragraphs)-1):
        prior_sentences_in_paragraph.append(' '.join(paragraph_sentences))
        if paragraphs[i] == paragraphs[i+1]:
            paragraph_sentences.append(sentences[i])
        else:
            paragraph_sentences = []
    prior_sentences_in_paragraph.append(' '.join(paragraph_sentences))
    return prior_sentences_in_paragraph


# tester code to estimate how long it will take for one sample
time_start = time.time()
print("sample: ", continue_one_sentence('hi there'))
print(time.time()-time_start)


if mode == "story_cloze":
    csv_filename = "cloze_test_val_winter2018_features_combined.csv"
else:
    csv_filename = "hippocorpus_paragraph_type_and_surprise_annotation_by_sentence_with_extracted.csv"

df = pd.read_csv(csv_filename)
paragraphs = df['paragraph']
sentences = df['sentence']

prior_sentences_in_paragraph = get_prior_sentences_in_paragraph(paragraphs, sentences)

df.insert(3, "prior_sentences_in_paragraph", prior_sentences_in_paragraph, allow_duplicates=True)

tnlg_generated_following_sentence = [continue_one_sentence(sentence) for sentence in tqdm(sentences)]

index = 5 if mode == "story_cloze" else 7

df.insert(index, "tnlg_generated_following_sentence", tnlg_generated_following_sentence, allow_duplicates=True)

df.to_csv(csv_filename.replace(".csv", "_tnlg_sentence.csv"))
