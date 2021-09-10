#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from nltk import sent_tokenize
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
import argparse

parser = argparse.ArgumentParser(description='call tnlg')
parser.add_argument("--src", default="surprise",
                              choices=["surprise", "story_cloze"])
args = parser.parse_args()

src = args.src

mode = "sbert" # "roberta"

if src == "story_cloze":
    filename = "cloze_test_val_winter2018_features_combined_tnlg_sentence.csv"
    output_filename = "story_cloze_sentence_cosine_similarity_updated_sbert.csv"
else:
    filename = "hippocorpus_paragraph_type_and_surprise_annotation_by_sentence_tnlg_sentence.csv"
    output_filename = "sentence_cosine_similarity_updated_sbert.csv"

df = pd.read_csv(filename)

tnlg = list(df['tnlg_generated_following_sentence'])

sentences = list(df['sentence'])
prior_sentences = list(df['prior_sentences_in_parapgraph'])

useful_indices = [i-1 for i in range(len(prior_sentences)) if isinstance(prior_sentences[i], str)]

def filter_indices(original, indices):
    return [original[i] for i in indices]
def get_first_sentence(text):
    if len(sent_tokenize(text)) > 0:
        return sent_tokenize(text)[0]
    return text

prior_sentences = filter_indices(sentences, useful_indices)
ground_sentences = filter_indices(sentences, [i+1 for i in useful_indices])

tnlg = filter_indices(tnlg, useful_indices)
tnlg = [get_first_sentence(text) for text in tnlg]

if mode == "roberta":
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    model.eval()

    def get_sentence_embedding(sentence):
        input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        with torch.no_grad():
            outputs = model(input_ids)
        last_hidden_states = outputs[0].squeeze(0) #(batch_size, input_len, embedding_size) But I need single vector for each sentence
        sentence_vector = torch.mean(last_hidden_states, axis=0)
        return sentence_vector.numpy()

elif mode == "sbert":
    model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
    model.eval()
    
    def get_sentence_embedding(sentence):
        embeddings = model.encode(sentence)
        return embeddings

def get_cosine_similarity(sentence0, sentence1):
    a = get_sentence_embedding(sentence0)
    b = get_sentence_embedding(sentence1)
    c = cosine_similarity(a.reshape(1, -1), Y=b.reshape(1, -1))
    return c.item()


cosine_similarities = [get_cosine_similarity(ground_sentences[i], tnlg[i]) for i in trange(len(tnlg))]

new_df = pd.DataFrame.from_dict({
                'prior_sentence':prior_sentences,
                'ground_sentence':ground_sentences,
                'tnlg_generated_next_sentence':tnlg,
                'cosine_similarities_sentence':cosine_similarities
            })

new_df.to_csv(output_filename)
