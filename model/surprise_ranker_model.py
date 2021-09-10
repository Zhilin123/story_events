#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
import numpy as np
import csv
import copy
from tqdm import tqdm, trange
import math
import os
import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, RobertaTokenizer,RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup
from surprise_ranker_dataset import SurpriseDataset
from torch import nn
import argparse
from sklearn import metrics
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='train_model')
parser.add_argument("--debug_mode", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--load_trained_model", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--lr", default="1e-4") #5e-5
parser.add_argument("--warmup_steps", default="1e2")
parser.add_argument("--config_name", default="default")
parser.add_argument("--inference_only", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Only performs generation without further training; load_trained model must be True")
parser.add_argument("--generation_name", default="",help="top_k-5, top_p-0.5, temperature-0.5, num_beams-5, original_tokens, original_spans")
parser.add_argument("--data_subset", default="all", choices=["all", "within_sentence", "not_within_sentence"])
parser.add_argument("--mode", default="head_reln_tail",choices=[
                                            "head_reln_tail", "head_tail_reln",
                                            "reln_head_tail", "reln_tail_head",
                                            "tail_reln_head", "tail_head_reln",
                                            "head", "reln", "tail"]) # dropped support for all_together
parser.add_argument("--model_name", default="roberta", choices=["bert", "roberta"])
parser.add_argument("--generate_train", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--train_dataset_filename", default="all_features_including_annotations_sentence_level.csv")
parser.add_argument("--custom_dataset_filename", default="hippocorpus_paragraph_type_and_surprise_annotation_by_sentence_with_extracted.csv")
parser.add_argument("--custom_save_filename", default="")
parser.add_argument("--generate_test", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--generate_custom", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--add_negative_samples", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--hippocorpus_category", default="recalled", choices=["recalled", "imagined", "retold","all"])
parser.add_argument("--glucose", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--structured", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--reverse", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--hippo_reverse", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--paragraph_level", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--only_roberta", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--only_features", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--loss_weighing", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--gru", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--gru_only", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--kl_div", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--gru_length_limit", type=int, default=30)
parser.add_argument("--feature_mode", default="features",choices=["features", "prior_features", "prior_confidence"])
parser.add_argument("--story_cloze", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--story_cloze_train_no_filter", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--remove_tnlg", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--remove_glucose_reverse", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--k_fold", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--total_folds", type=int, default=10, choices=[5, 10])
parser.add_argument("--fold_number", type=int, default=1, choices=[1,2,3,4,5,6,7,8,9,10])
parser.add_argument("--save_model", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--remove_equal_validation", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--fast_classifier_weights", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--infer_story_cloze", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--story_cloze_train_dataset_filename", default="all_features_including_annotations_story_cloze_with_prior_confidence.csv")

args = parser.parse_args()

debug_mode = args.debug_mode
load_trained_model = args.load_trained_model
epochs = args.epochs
max_epochs = args.max_epochs
learning_rate = float(args.lr)
warmup_steps = float(args.warmup_steps)
config_name = args.config_name
inference_only = args.inference_only
generation_name = args.generation_name
data_subset = args.data_subset
mode = args.mode #"all_together" #
model_name = args.model_name
generate_train = args.generate_train
batch_size = args.batch_size
train_dataset_filename = args.train_dataset_filename
generate_test = args.generate_test
generate_custom = args.generate_custom
eval_batch_size = min(args.eval_batch_size, batch_size)
add_negative_samples = args.add_negative_samples
hippocorpus_category = args.hippocorpus_category
#hippocorpus_extracted = args.hippocorpus_extracted
glucose = args.glucose
custom_dataset_filename = args.custom_dataset_filename
custom_save_filename = args.custom_save_filename
structured = args.structured
reverse = args.reverse
hippo_reverse = args.hippo_reverse
paragraph_level = args.paragraph_level


sample_every = 300 if not debug_mode else 1

# Set the seed value all over the place to make this reproducible.
seed_val = args.random_seed

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

if not os.path.exists(config_name+"/"):
    os.makedirs(config_name+"/")

checkpointer_name = "{}/discriminator_pytorch_model.pth".format(config_name)
best_checkpointer_name = "{}/discriminator_pytorch_model_best.pth".format(config_name)
training_stats_filename = "{}/discriminator_training_stats.csv".format(config_name)
eval_stats_filename = "{}/discriminator_eval_stats.csv".format(config_name)




eval_every = 0.25

epsilon = 1e-8

adjust_sample_weight = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_labels = 9*(int(reverse)+1)+int(add_negative_samples)+int(glucose)

class SurpriseClassifier(nn.Module):

    def __init__(self, bert_model):
        super().__init__()

        dim = 43
        if args.remove_tnlg:
            dim -= 1
        if args.remove_glucose_reverse:
            dim -= 10
        predict_dim = 2 if args.story_cloze else 3
        self.bert_model = bert_model
        self.classifier = nn.Linear(dim, predict_dim)
        self.pooler = nn.Linear(768, 1)
        self.bert_classifier = nn.Linear(768, predict_dim)
        self.tanh = nn.Tanh()
        self.feature_dense = nn.Linear(dim, dim)
        self.combined_classifier = nn.Linear(768+dim, predict_dim)
        self.combine_bert_and_feature = nn.Linear(predict_dim*2, predict_dim)
        self.combine_bert_and_feature.weight.data = torch.cat([torch.eye(predict_dim),
                                                               torch.eye(predict_dim)],dim=1)
        self.gru = nn.GRU(dim, dim, batch_first=True)

        self.loss_fct = nn.CrossEntropyLoss() if not args.kl_div else nn.KLDivLoss()

    def forward(self, input_ids, attention_mask, features, sep_positions, label=None):

        hidden_states = self.bert_model(input_ids, attention_mask)


        if args.only_roberta:
            sep_token_states = hidden_states[1]
            output = self.bert_classifier(sep_token_states)

        if args.gru or args.gru_only:
            if isinstance(features, torch.Tensor):
                if len(features.size()) == 2:
                    features = features.unsqueeze(1)
                gru_output, hn = self.gru(features)
                gru_output = gru_output[:,-1,:]
            else:
                all_gru_output, hn = self.gru(features)
                gru_tensor_output, length = torch.nn.utils.rnn.pad_packed_sequence(all_gru_output, batch_first=True)
                last_element_gru = [gru_tensor_output[i, length[i].item()-1, :] for i in range(gru_tensor_output.size(0))]
                gru_output = torch.stack(last_element_gru)

            features = self.classifier(gru_output)
            if args.gru_only:
                output = features
            else:
                sep_token_states = hidden_states[1]
                bert_output = self.bert_classifier(sep_token_states)
                output = self.combine_bert_and_feature(torch.cat([features, bert_output], dim=1))

        if label is not None:
            if not args.kl_div:
                loss = self.loss_fct(output, label.view(-1))
            else:
                log_softmax_output = nn.LogSoftmax(dim=1)(output)
                loss = self.loss_fct(log_softmax_output, label)
            return loss, output
        else:
            return output

if model_name == "bert":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained("bert-base-uncased", num_labels=num_labels)
elif model_name == "roberta":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    bert_model = RobertaModel.from_pretrained("roberta-base", num_labels=num_labels)

model = SurpriseClassifier(bert_model)


tokenizer.add_special_tokens({
                              'pad_token': '[PAD]',
                              'bos_token': '[CLS]',
                              'eos_token': '[PAD]',
                              'sep_token': '[SEP]'
                              })

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
model.bert_model.resize_token_embeddings(len(tokenizer))




if not args.fast_classifier_weights:
    optimizer = AdamW(
                      model.parameters(),
                      lr = learning_rate,
                      eps = epsilon
                    )
else:

    optimizer = AdamW(
                      [
                         {'params': model.bert_model.parameters()},
                         {'params': model.classifier.parameters(), 'lr': 1e-3},
                         {'params': model.gru.parameters(), 'lr': 1e-3},
                         {'params': model.bert_classifier.parameters()},
                         {'params': model.combine_bert_and_feature.parameters()},
                      ],
                      lr = learning_rate,
                      eps = epsilon
                    )

def my_collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    if args.kl_div:
        labels = torch.FloatTensor([item[2] for item in batch])
    else:
        labels = torch.LongTensor([item[2] for item in batch])
    sep_positions = torch.stack([item[4] for item in batch])

    if args.feature_mode == "features":
        features = torch.stack([item[3] for item in batch])
    else:
        features = torch.nn.utils.rnn.pack_padded_sequence(
                torch.stack([torch.cat((item[3], torch.zeros((args.gru_length_limit-item[3].size(0),item[3].size(1)))), dim=0) for item in batch]),
                [item[3].size(0) for item in batch],
                batch_first=True,
                enforce_sorted=False
            )

    return (input_ids, masks, labels, features, sep_positions)

if load_trained_model:
    if inference_only:
        checkpoint = torch.load(best_checkpointer_name, map_location=device)
    else:
        checkpoint = torch.load(checkpointer_name, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    last_finished_epoch = checkpoint['epoch']
    starting_epoch = last_finished_epoch + 1

    with open(training_stats_filename) as f:
        training_stats = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]

    print("starting_epoch: ", starting_epoch)

else:
    starting_epoch = 0
    training_stats = []


model = model.to(device)

dataset_class = SurpriseDataset

if args.k_fold and args.generate_test:
    raise ValueError("k-fold cross validation only allows val set")

train_dataset = dataset_class(train_dataset_filename, tokenizer,
                              debug_mode=(debug_mode or inference_only),
                              data_subset="trn",
                              paragraph_level=paragraph_level,
                              prior_features_length_limit=args.gru_length_limit,
                              feature_mode=args.feature_mode,
                              story_cloze=args.story_cloze,
                              no_filter=args.story_cloze_train_no_filter,
                              kl_div=args.kl_div,
                              remove_glucose_reverse=args.remove_glucose_reverse,
                              remove_tnlg=args.remove_tnlg,
                              k_fold=args.k_fold,
                              fold_number=args.fold_number,
                              total_folds=args.total_folds)

if args.infer_story_cloze:
    val_dataset = dataset_class(args.story_cloze_train_dataset_filename,tokenizer,
                                debug_mode=debug_mode,
                                data_subset="all",
                                paragraph_level=paragraph_level,
                                prior_features_length_limit=args.gru_length_limit,
                                feature_mode=args.feature_mode,
                                story_cloze=True,
                                kl_div=False,
                                remove_glucose_reverse=args.remove_glucose_reverse,
                                remove_tnlg=args.remove_tnlg,
                                k_fold=args.k_fold,
                                fold_number=args.fold_number,
                                total_folds=args.total_folds,
                                remove_equal=args.remove_equal_validation)
else:
    val_dataset = dataset_class(train_dataset_filename,tokenizer,
                                debug_mode=debug_mode,
                                data_subset=("tst" if generate_test else "dev"),
                                paragraph_level=paragraph_level,
                                prior_features_length_limit=args.gru_length_limit,
                                feature_mode=args.feature_mode,
                                story_cloze=args.story_cloze,
                                kl_div=args.kl_div,
                                remove_glucose_reverse=args.remove_glucose_reverse,
                                remove_tnlg=args.remove_tnlg,
                                k_fold=args.k_fold,
                                fold_number=args.fold_number,
                                total_folds=args.total_folds,
                                remove_equal=args.remove_equal_validation)



if args.loss_weighing and not args.kl_div:
    train_dataset.loss_weights.to(device)
    val_dataset.loss_weights.to(device)
    model.loss_fct = nn.CrossEntropyLoss(weight=train_dataset.loss_weights)

print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(val_dataset)))


train_dataloader = DataLoader(
            train_dataset,
            sampler = SequentialSampler(train_dataset), #RandomSampler(train_dataset),
            batch_size = batch_size,
            collate_fn=my_collate_fn
        )


validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size,
            collate_fn=my_collate_fn
        )

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = warmup_steps,
                                            num_training_steps = total_steps)

if load_trained_model:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# save
def save(model, optimizer, scheduler,checkpointer_name, epoch):
    output_dir = "/".join(checkpointer_name.split("/")[:-1]) + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    # save
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict':scheduler.state_dict(),
        'epoch':epoch
    }, checkpointer_name)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def precision_recall_fscore(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1 = (2 * precision * recall) / (precision+recall+ 1e-12)
    return precision, recall, f1

def get_p_r_f1(report, field):
    if field in report:
        return [report[field][i] for i in ['precision', 'recall', 'f1-score']]
    return [0,0,0]

def get_metrics(y_true, y_pred):
    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    return [get_p_r_f1(report, field) for field in ['weighted avg', '0', '1', '2']]

def calculate_token_wise_accuracy(b_logits, b_input_ids, b_labels):

    b_logits = torch.nn.Softmax(dim=1)(b_logits) #[:, 1])[highest_prob_index].item()

    all_tokens = []

    for i in range(b_input_ids.size(0)):
        sep_token_positions = [k for k in range(len(b_input_ids[i])) if b_input_ids[i][k] == tokenizer.sep_token_id]
        prior_sentence = b_input_ids[i, 1:sep_token_positions[0]]
        interested_sentence = b_input_ids[i, sep_token_positions[0]+1:sep_token_positions[1]]
        highest_prob_index = torch.argmax(b_logits[i, :])
        confidence = b_logits[i, :].tolist()
        predicted_reln = highest_prob_index.item()
        if args.kl_div and not args.infer_story_cloze:
            ground_reln = torch.argmax(b_labels[i,:]).item()
        else:
            ground_reln = b_labels[i].item()
        all_tokens.append([prior_sentence,
                           interested_sentence,
                           ground_reln, predicted_reln,
                           confidence])

    return all_tokens

def decode_and_save_all_tokens(all_tokens, epoch_equivalent):
    tokens_to_words = {}
    formatted_decoded_tokens = []
    for one_set_of_tokens in all_tokens:
        for one_token in one_set_of_tokens:
            one_decoded_set_of_tokens = []
            for one_small_token in one_token[:-3]:
                one_token_list = tuple([i for i in one_small_token]) #if 0 <= i < len(tokenizer)
                if one_token_list not in tokens_to_words:
                    try:
                        tokens_to_words[one_token_list] = tokenizer.decode(one_token_list)
                    except TypeError:
                        print("cannot be decoded: ", one_token_list)
                        tokens_to_words[one_token_list] = ''
                one_decoded_set_of_tokens.append(tokens_to_words[one_token_list])
            #this is confidence
            one_decoded_set_of_tokens += one_token[-3:]

            prior_sentence, interested_sentence, ground_reln, predicted_reln, confidence = one_decoded_set_of_tokens

            formatted_decoded_tokens.append({
                        "prior_sentence":prior_sentence,
                        "interested_sentence":interested_sentence,
                        "ground_reln":ground_reln,
                        "predicted_reln":predicted_reln,
                        "confidence":confidence
                    })

    df = pd.DataFrame(data=formatted_decoded_tokens)
    if custom_save_filename:
        df.to_csv("{}/{}".format(config_name, custom_save_filename))
    elif generate_train:
        df.to_csv("{}/surprise_discriminator_train_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    elif generate_custom:
        df.to_csv("{}/surprise_discriminator_custom_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    elif generate_test:
        df.to_csv("{}/surprise_discriminator_test_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    else:
        df.to_csv("{}/surprise_discriminator_eval_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))

    return None

def save_json(docs, output_filename):
    with open(output_filename, "w") as write_file:
        json.dump(docs, write_file)

def save_feature_weights():
    features = {
                "combine_bert_and_feature":model.combine_bert_and_feature.weight.data.cpu().tolist(),
                "classifier":model.classifier.weight.data.cpu().tolist(),
                "bert_classifier":model.bert_classifier.weight.data.cpu().tolist()
            }

    output_filename = "{}/best_features.json".format(config_name)
    save_json(features, output_filename)

def eval_once(epoch_equivalent):

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0
    total_eval_ppl = 0

    all_tokens = []
    ground_labels = []
    predicted_labels = []
    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(validation_dataloader), position=0, leave=True):

        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_features = batch[3].to(device)
        b_sep_positions = batch[4].to(device)

        with torch.no_grad():

            if not args.infer_story_cloze:
                outputs = model(b_input_ids, b_masks, b_features, b_sep_positions, label=b_labels)
                loss = outputs[0]
                b_logits = outputs[1]
                batch_loss = loss.item()
            else:
                b_logits = model(b_input_ids, b_masks, b_features, b_sep_positions, label=None)
                batch_loss = 0


        total_eval_loss += batch_loss
        total_eval_ppl += 2**batch_loss
        ground_labels.append(b_labels)
        one_predicted_label = torch.argmax(b_logits, dim=1)
        predicted_labels.append(one_predicted_label)

        tokens = calculate_token_wise_accuracy(b_logits, b_input_ids, b_labels)
        all_tokens.append(tokens)

        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, total_epochs))
            print(' Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}'.format(step, len(validation_dataloader), batch_loss, elapsed))


    ground_labels = torch.cat(ground_labels, dim=0).cpu()

    if args.kl_div and not args.infer_story_cloze:
        ground_labels = torch.argmax(ground_labels, dim=1)

    predicted_labels = torch.cat(predicted_labels, dim=0).cpu()


    avg_eval_loss = total_eval_loss / len(validation_dataloader)
    avg_eval_ppl = total_eval_ppl / len(validation_dataloader)

    save_tokens_name = generation_name if inference_only else epoch_equivalent

    overall_metrics, noevent_metrics, expected_metrics, surprising_metrics = get_metrics(ground_labels, predicted_labels)
    decode_and_save_all_tokens(all_tokens, save_tokens_name)


    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_eval_loss))
    print("  Validation took: {:}".format(validation_time))

    if inference_only:

        if os.path.exists(eval_stats_filename):
            with open(eval_stats_filename) as f:
                eval_stats = [{k: v for k, v in row.items()}
                    for row in csv.DictReader(f, skipinitialspace=True)]
        else:
            eval_stats = []

        eval_stats.append({
                'Generation Name': generation_name,
                'config_name':config_name,
                'Valid. Loss': avg_eval_loss,
                'Validation Time': validation_time,
                'precision':overall_metrics[0],
                'recall':overall_metrics[1],
                'f1': overall_metrics[2],
                'precision_noevent':noevent_metrics[0],
                'recall_noevent':noevent_metrics[1],
                'f1_noevent': noevent_metrics[2],
                'precision_expected':expected_metrics[0],
                'recall_expected': expected_metrics[1],
                'f1_expected': expected_metrics[2],
                'precision_surprising': surprising_metrics[0],
                'recall_surprising': surprising_metrics[1],
                'f1_surprising': surprising_metrics[2]
            })

        pd.set_option('precision', 5)
        df_stats = pd.DataFrame(data=eval_stats)
        df_stats = df_stats.set_index('Generation Name')
        df_stats.to_csv(eval_stats_filename)

        print("total/noEvent/expected/surprising F1: {}".format(' & '.join([
            str(round(eval_stats[-1][factor],4))
                for factor in ['f1', 'f1_noevent', 'f1_expected', 'f1_surprising']
            ] + ['\\\\'])))

        raise ValueError




    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_equivalent,
            'config_name':config_name,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_eval_loss,
            'Training Time': training_time,
            'Validation Time': validation_time,
            'precision':overall_metrics[0],
            'recall':overall_metrics[1],
            'f1': overall_metrics[2],
            'precision_noevent':noevent_metrics[0],
            'recall_noevent':noevent_metrics[1],
            'f1_noevent': noevent_metrics[2],
            'precision_expected':expected_metrics[0],
            'recall_expected': expected_metrics[1],
            'f1_expected': expected_metrics[2],
            'precision_surprising': surprising_metrics[0],
            'recall_surprising': surprising_metrics[1],
            'f1_surprising': surprising_metrics[2]
        }
    )


    all_f1s = [i['f1'] for i in training_stats]
    best_f1_position = np.argmax(all_f1s)


    print("best so far by f1 (all)")
    for factor in ['epoch', 'f1', 'precision', 'recall', 'Valid. Loss']: #, 'avg_eval_all_correct'
        print("{}: {}".format(factor, round(training_stats[best_f1_position][factor],4)))

    print("total/noEvent/expected/surprising F1: {}".format(' & '.join([
            str(round(training_stats[best_f1_position][factor],4))
                for factor in ['f1', 'f1_noevent', 'f1_expected', 'f1_surprising']
            ] + ['\\\\'])))

    pd.set_option('precision', 5)
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    df_stats.to_csv(training_stats_filename)
    if not inference_only:
        if int(epoch_equivalent) == round(epoch_equivalent,2):
            pass
            #if args.save_model:
            #   save(model, optimizer,scheduler, checkpointer_name, epoch_i)
        if training_stats[best_f1_position]['epoch'] == epoch_equivalent:
            if args.save_model:
                save(model, optimizer,scheduler, best_checkpointer_name, epoch_i)
            save_feature_weights()



total_t0 = time.time()

model = model.to(device)

total_epochs = min(max_epochs, epochs+starting_epoch)

for epoch_i in range(starting_epoch, total_epochs):

    if not inference_only:
        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, total_epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0
        total_train_ppl = 0

        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):

            model.zero_grad()

            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_features = batch[3].to(device)
            b_sep_positions = batch[4].to(device)

            outputs = model(b_input_ids, b_masks, b_features, b_sep_positions, label=b_labels)
            loss = outputs[0]
            b_logits = outputs[1]

            batch_loss = loss.item()
            total_train_loss += batch_loss
            total_train_ppl += 2**batch_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs+starting_epoch))
                print(' Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}'.format(step, len(train_dataloader), batch_loss, elapsed))
                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()

            if (step+1) % int(eval_every * len(train_dataloader)) == 0 and step != 0:
                avg_train_loss = total_train_loss / (step+1)
                avg_train_ppl = total_train_ppl/ (step+1)
                training_time = format_time(time.time() - t0)

                eval_once(epoch_i + (step+1)/len(train_dataloader))
                model.train()

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))


    else:
        eval_once(epoch_i + 1)
        break


print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
