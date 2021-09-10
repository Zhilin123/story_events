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
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer,RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer
from atomic_and_glucose_dataset import AtomicDataset, HippoAtomicDataset, GlucoseDataset, HippoGlucoseDataset, StoryClozeAtomicDataset, StoryClozeGlucoseDataset
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
parser.add_argument("--mode", default="head_reln_tail") 
parser.add_argument("--model_name", default="roberta", choices=["bert", "roberta"])
parser.add_argument("--generate_train", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--train_dataset_filename", default="v4_atomic_all_agg.csv")
parser.add_argument("--custom_dataset_filename", default="hippocorpus_paragraph_type_and_surprise_annotation_by_sentence_with_extracted.csv")
parser.add_argument("--custom_save_filename", default="")
parser.add_argument("--generate_test", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--generate_custom", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--add_negative_samples", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--hippocorpus_category", default="all", choices=["recalled", "imagined", "retold","all"])
#parser.add_argument("--hippocorpus_extracted", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--glucose", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--structured", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--reverse", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--hippo_reverse", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--paragraph_level", type=lambda x: (str(x).lower() == 'true'), default=False)

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
seed_val = 42

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

if model_name == "bert":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
elif model_name == "roberta":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)


if glucose:
    column_ending_with = "specificStructured" if structured else "specificNL"
    relations = ["{}_{}".format(i,column_ending_with) for i in range(1,11)] + ["xNone"]

    if structured:
        relations += ["[SUBJ]","[VERB]","[PRP1]","[OBJ1]", "[PRP2]","[OBJ2]"]
else:
    relations = ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent",
                 "xNeed", "xReact", "xWant"]

    if reverse:
        relations_old = copy.copy(relations)
        relations += [i+"-r" for i in relations_old]

    if add_negative_samples:
        relations += ["xNone"]

    relations += ["PersonX", "PersonY", "___"]

print("relations: ", relations)

tokenizer.add_special_tokens({'pad_token': '[PAD]',
                              'bos_token': '[CLS]',
                              'eos_token': '[PAD]',
                              "additional_special_tokens":["[SEP]", "[RELN]"] + relations
                              })

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
model.resize_token_embeddings(len(tokenizer))

optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

if load_trained_model:
    #model_state_dict = torch.load(bin_name)
    #model.load_state_dict(model_state_dict)
    # load
    if inference_only:
        checkpoint = torch.load(best_checkpointer_name, map_location=device)
    else:
        checkpoint = torch.load(checkpointer_name, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
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

dataset_class = GlucoseDataset if glucose else AtomicDataset
if "hippocorpus" in custom_dataset_filename:
    hippo_dataset_class = HippoGlucoseDataset if glucose else HippoAtomicDataset
else:
    hippo_dataset_class = StoryClozeGlucoseDataset if glucose else StoryClozeAtomicDataset

train_dataset = dataset_class(train_dataset_filename, tokenizer,
                              debug_mode=(debug_mode or inference_only),
                              data_subset="trn", add_negative_samples=add_negative_samples,
                              structured=structured, reverse=reverse)

if generate_custom:
    val_dataset = hippo_dataset_class(custom_dataset_filename, tokenizer,
                                     debug_mode=debug_mode, data_subset=hippocorpus_category,
                                     add_negative_samples=add_negative_samples,
                                     structured=structured, reverse=hippo_reverse,
                                     paragraph_level=paragraph_level)
else:
    val_dataset = dataset_class(train_dataset_filename,tokenizer,
                                debug_mode=debug_mode,
                                data_subset=("tst" if generate_test else "dev"),
                                add_negative_samples=add_negative_samples,
                                structured=structured, reverse=reverse,
                                paragraph_level=paragraph_level)


print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(val_dataset)))


train_dataloader = DataLoader(
            train_dataset,
            sampler = SequentialSampler(train_dataset), #RandomSampler(train_dataset),
            batch_size = batch_size
        )


validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
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


id_of_special_tokens = tokenizer.convert_tokens_to_ids(relations+ ['[CLS]', '[RELN]', '[SEP]', '[PAD]'])

head_token_id = tokenizer.convert_tokens_to_ids("[CLS]")
reln_token_id = tokenizer.convert_tokens_to_ids("[RELN]")
tail_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
eos_token_id = tokenizer.eos_token_id


series = [head_token_id, reln_token_id, tail_token_id, eos_token_id]

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def get_start_and_end_token_ids(category="head", mode="head_reln_tail"):
    if category == "head":
        return head_token_id, reln_token_id
    elif category == "reln":
        return reln_token_id, tail_token_id
    elif category == "tail":
        return tail_token_id, eos_token_id

def get_one_span(tokens, category="head"):
    start_token_id, end_token_id = get_start_and_end_token_ids(category=category, mode=mode)
    start = 0
    end = len(tokens) #exclusive
    start_found = False
    end_found = False
    for i in range(len(tokens)):
        if tokens[i] == start_token_id:
            start = i
            start_found = True
        if tokens[i] == end_token_id:
            end = i
            end_found = True
            break
    return tokens[start+1:end]#, start_found, end_found, start, end

def get_head_reln_tail_tokens(tokens):

    start_pos = 0
    end_pos = len(tokens)-1
    for j in range(tokens.size(-1)):
        if tokens[j] == series[0]:
            start_pos = j
        if tokens[j] == tokenizer.eos_token_id:
            end_pos = j

    relevant_tokens = tokens[start_pos:end_pos+1]

    head_tokens = get_one_span(relevant_tokens, category="head")
    reln_tokens = get_one_span(relevant_tokens, category="reln")
    tail_tokens = get_one_span(relevant_tokens, category="tail")

    return [head_tokens, reln_tokens, tail_tokens]

def calculate_token_wise_accuracy(b_logits, b_input_ids, b_labels):

    # b_labels --> tensor (batch_size,) # needs to have one correct one

    # b_logits --> tensor (batch_size, n_labels)
    # b_input_ids --> tensor (batch_size, seq_len) --> just take random one)

    # process ground truth sentence
    '''
    location_of_first_head_token = [k for k in range(len(b_input_ids[0])) if b_input_ids[0][k] == series[0]][0]

    location_of_first_bos_token = [k for k in range(len(b_input_ids[0])) if b_input_ids[0][k] == tokenizer.bos_token_id][0]

    ground_sentence = b_input_ids[0][location_of_first_bos_token+1:location_of_first_head_token]

    ground_truth_tokens = b_ground_truth_tokens[0]

    '''

    #confidence = torch.nn.Softmax(dim=0)(b_logits[:, 1])[highest_prob_index].item()

    all_tokens = []

    if not generate_custom:
        b_logits = F.softmax(b_logits, dim=-1)

    for i in range(b_logits.size(0)):
        # this is batch size
        label = b_labels[i].item()
        #highest_prob_index = b_logits[i, label]
        #predicted_tokens = b_input_ids[highest_prob_index]
        #ground_head, ground_reln, ground_tail = get_head_reln_tail_tokens(ground_truth_tokens)
        predicted_head, predicted_reln, predicted_tail = get_head_reln_tail_tokens(b_input_ids[i])
        highest_prob_index = torch.argmax(b_logits[i, :])
        if not generate_custom:
            confidence = b_logits[i, highest_prob_index].item()
        else:
            confidence = b_logits[i, :].tolist()

        predicted_reln = [id_of_special_tokens[highest_prob_index]]
        ground_reln = [id_of_special_tokens[label]]
        all_tokens.append([predicted_head, predicted_head,
                           ground_reln, predicted_reln,
                           predicted_tail, predicted_tail,
                           ground_reln, confidence])

    return all_tokens




def precision_recall_fscore(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1 = (2 * precision * recall) / (precision+recall+ 1e-12)
    return precision, recall, f1


def decode_and_save_all_tokens(all_tokens, epoch_equivalent):
    tokens_to_words = {}
    formatted_decoded_tokens = []
    for one_set_of_tokens in all_tokens:
        for one_token in one_set_of_tokens:
            one_decoded_set_of_tokens = []
            for one_small_token in one_token[:-1]:
                one_token_list = tuple([i for i in one_small_token]) #if 0 <= i < len(tokenizer)
                if one_token_list not in tokens_to_words:
                    try:
                        tokens_to_words[one_token_list] = tokenizer.decode(one_token_list)
                    except TypeError:
                        print("cannot be decoded: ", one_token_list)
                        tokens_to_words[one_token_list] = ''
                one_decoded_set_of_tokens.append(tokens_to_words[one_token_list])
            #this is confidence
            one_decoded_set_of_tokens.append(one_token[-1])

            ground_head, predicted_head, ground_reln, predicted_reln, ground_tail, predicted_tail, ground_sentence, confidence = one_decoded_set_of_tokens
            formatted_decoded_tokens.append({
                        "ground_head":ground_head,
                        "predicted_head":predicted_head,
                        "ground_reln":ground_reln,
                        "predicted_reln":predicted_reln,
                        "ground_tail":ground_tail,
                        "predicted_tail":predicted_tail,
                        "ground_sentence":ground_sentence,
                        "confidence":confidence
                    })

    all_ground_head = [i["ground_head"] for i in formatted_decoded_tokens]
    all_predicted_head = [i["predicted_head"] for i in formatted_decoded_tokens]
    all_ground_reln = [i["ground_reln"] for i in formatted_decoded_tokens]
    all_predicted_reln = [i["predicted_reln"] for i in formatted_decoded_tokens]
    all_ground_tail = [i["ground_tail"] for i in formatted_decoded_tokens]
    all_predicted_tail = [i["predicted_tail"] for i in formatted_decoded_tokens]

    all_ground = [all_ground_head[i] + all_ground_reln[i] + all_ground_tail[i] for i in range(len(all_ground_head))]
    all_predicted = [all_predicted_head[i] + all_predicted_reln[i] + all_predicted_tail[i] for i in range(len(all_ground_head))]

    precision, recall, f1 = precision_recall_fscore(all_ground, all_predicted)

    precision_head, recall_head, f1_head= precision_recall_fscore(all_ground_head, all_predicted_head)
    precision_reln, recall_reln, f1_reln= precision_recall_fscore(all_ground_reln, all_predicted_reln)
    precision_tail, recall_tail, f1_tail = precision_recall_fscore(all_ground_tail, all_predicted_tail)

    df = pd.DataFrame(data=formatted_decoded_tokens)
    if custom_save_filename:
        df.to_csv("{}/{}".format(config_name, custom_save_filename))
    elif generate_train:
        df.to_csv("{}/discriminator_train_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    elif generate_custom:
        df.to_csv("{}/discriminator_custom_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    elif generate_test:
        df.to_csv("{}/discriminator_test_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    else:
        df.to_csv("{}/discriminator_eval_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))

    return precision, recall, f1, \
           precision_head, recall_head, f1_head,\
           precision_reln, recall_reln, f1_reln, \
           precision_tail, recall_tail, f1_tail,

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

#    total_eval_all_correct = 0
#    total_eval_average_correct = 0
#    total_eval_all_head_correct = 0
#    total_eval_average_head_correct = 0
#    total_eval_all_reln_correct = 0
#    total_eval_average_reln_correct = 0
#    total_eval_all_tail_correct = 0
#    total_eval_average_tail_correct = 0

    all_tokens = []

    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(validation_dataloader), position=0, leave=True):

        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)
        #b_ground_truth_tokens = batch[3].to(device)

#        b_sample_weights = batch[3]
#        b_generate_input_ids = batch[4].to(device)
#        b_generate_attn_masks = batch[5].to(device)

        with torch.no_grad():

            outputs = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_masks,
                             labels=b_labels)

            loss = outputs[0]
            b_logits = outputs[1]


        batch_loss = loss.item()
        total_eval_loss += batch_loss
        total_eval_ppl += 2**batch_loss

#        if need_generation:
#            output_undecoded = generate(b_generate_input_ids, b_generate_attn_masks)
#            #print(" sentence: ", tokenizer.decode(output_undecoded))
#            # save the sentences themselves and use the bert model separately
#            #raise ValueError
#
##            generated_tokens = get_head_reln_tail_from_tokens(output_undecoded)
##            value_embeddings = get_embeddings_from_tokens(generated_tokens)
##            print(torch.stack(value_embeddings).size())
##            ds_embeddings = get_ds_embeddings()
##            print(torch.stack(ds_embeddings).size())
##            raise ValueError
#
#        else:
#            output_undecoded = torch.argmax(b_logits, axis=-1)
        #print(b_logits.size(0), eval_batch_size)
#        if b_logits.size(0) == eval_batch_size:
        tokens = calculate_token_wise_accuracy(b_logits, b_input_ids, b_labels) #scores,
        all_tokens.append(tokens)
#        else:
#            for i in range(0, b_logits.size(0), eval_batch_size):
#                tokens = calculate_token_wise_accuracy(b_logits[i:i+eval_batch_size, :], b_input_ids[i:i+eval_batch_size, :], b_ground_truth_tokens[i:i+eval_batch_size, :]) #scores,
#                all_tokens.append(tokens)

         # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, total_epochs))
            print(' Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}'.format(step, len(validation_dataloader), batch_loss, elapsed))
#            print(' Average_correct: {} All_correct: {}'.format(total_eval_average_correct/(step+1),total_eval_all_correct/(step+1)))
#            print(' All_head_correct: {} '.format(total_eval_all_head_correct/(step+1)))
#            print(' All_reln_correct: {} '.format(total_eval_all_reln_correct/(step+1)))
#            print(' All_tail_correct: {} '.format(total_eval_all_tail_correct/(step+1)))
            print(' ppl {} '.format(total_eval_ppl/(step+1)))



    avg_eval_loss = total_eval_loss / len(validation_dataloader)
    avg_eval_ppl = total_eval_ppl / len(validation_dataloader)

#    avg_eval_all_correct = total_eval_all_correct / len(validation_dataloader)
#    #avg_eval_average_correct = total_eval_average_correct / len(validation_dataloader)
#
#    avg_eval_all_head_correct = total_eval_all_head_correct / len(validation_dataloader)
#    #avg_eval_average_head_correct = total_eval_average_head_correct / len(validation_dataloader)
#    avg_eval_all_reln_correct = total_eval_all_reln_correct / len(validation_dataloader)
#    #avg_eval_average_reln_correct = total_eval_average_reln_correct / len(validation_dataloader)
#    avg_eval_all_tail_correct = total_eval_all_tail_correct / len(validation_dataloader)
#    #avg_eval_average_tail_correct = total_eval_average_tail_correct / len(validation_dataloader)
#
    save_tokens_name = generation_name if inference_only else epoch_equivalent

    precision, recall, f1, \
    precision_head, recall_head, f1_head,\
    precision_reln, recall_reln, f1_reln, \
    precision_tail, recall_tail, f1_tail = decode_and_save_all_tokens(all_tokens, save_tokens_name)

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
                'Valid. Loss': avg_eval_loss,
                'Validation Time': validation_time,
                'avg_eval_ppl':avg_eval_ppl,
#                'avg_eval_all_correct':avg_eval_all_correct,
#                'avg_eval_average_correct': avg_eval_average_correct,
                'precision':precision,
                'recall':recall,
                'f1': f1,
#                'avg_eval_all_head_correct':avg_eval_all_head_correct,
#                'avg_eval_average_head_correct':avg_eval_average_head_correct,
                'precision_head':precision_head,
                'recall_head':recall_head,
                'f1_head': f1_head,
#                'avg_eval_all_reln_correct':avg_eval_all_reln_correct,
#                'avg_eval_average_reln_correct':avg_eval_average_reln_correct,
                'precision_reln':precision_reln,
                'recall_reln':recall_reln,
                'f1_reln': f1_reln,
#                'avg_eval_all_tail_correct':avg_eval_all_tail_correct,
#                'avg_eval_average_tail_correct':avg_eval_average_tail_correct,
                'precision_tail':precision_tail,
                'recall_tail':recall_tail,
                'f1_tail': f1_tail,
            })

        pd.set_option('precision', 5)
        df_stats = pd.DataFrame(data=eval_stats)
        df_stats = df_stats.set_index('Generation Name')
        df_stats.to_csv(eval_stats_filename)

        raise ValueError


    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_equivalent,
            'config_name':config_name,
            'Training Loss': avg_train_loss,
            'avg_train_ppl':avg_train_ppl,
#            'avg_train_all_correct':avg_train_all_correct,
#            'avg_train_average_correct': avg_train_average_correct,
#            'avg_train_all_head_correct':avg_train_all_head_correct,
#            'avg_train_average_head_correct':avg_train_average_head_correct,
#            'avg_train_all_reln_correct':avg_train_all_reln_correct,
#            'avg_train_average_reln_correct':avg_train_average_reln_correct,
#            'avg_train_all_tail_correct':avg_train_all_tail_correct,
#            'avg_train_average_tail_correct':avg_train_average_tail_correct,
            'Valid. Loss': avg_eval_loss,
            'Training Time': training_time,
            'Validation Time': validation_time,
            'avg_eval_ppl':avg_eval_ppl,
#            'avg_eval_all_correct':avg_eval_all_correct,
#            'avg_eval_average_correct': avg_eval_average_correct,
#            'avg_eval_all_head_correct':avg_eval_all_head_correct,
#            'avg_eval_average_head_correct':avg_eval_average_head_correct,
#            'avg_eval_all_reln_correct':avg_eval_all_reln_correct,
#            'avg_eval_average_reln_correct':avg_eval_average_reln_correct,
#            'avg_eval_all_tail_correct':avg_eval_all_tail_correct,
#            'avg_eval_average_tail_correct':avg_eval_average_tail_correct,
            'precision':precision,
            'recall':recall,
            'f1': f1,
            'precision_head':precision_head,
            'recall_head':recall_head,
            'f1_head': f1_head,
            'precision_reln':precision_reln,
            'recall_reln':recall_reln,
            'f1_reln': f1_reln,
            'precision_tail':precision_tail,
            'recall_tail':recall_tail,
            'f1_tail': f1_tail,
        }
    )


    all_f1s = [i['f1'] for i in training_stats]
    best_f1_position = np.argmax(all_f1s)

    if training_stats[best_f1_position]["f1"] == 0:
        all_f1s = [i['f1_head'] + i['f1_reln'] + i['f1_tail'] - i['Valid. Loss'] for i in training_stats]
        best_f1_position = np.argmax(all_f1s)


    print("best so far by f1 (all)")
    for factor in ['epoch', 'f1', 'precision', 'recall', 'Valid. Loss']: #, 'avg_eval_all_correct'
        print("{}: {}".format(factor, training_stats[best_f1_position][factor]))

    pd.set_option('precision', 5)
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    df_stats.to_csv(training_stats_filename)
    if not inference_only:
        if int(epoch_equivalent) == round(epoch_equivalent,2):
            save(model, optimizer,scheduler, checkpointer_name, epoch_i)
        if training_stats[best_f1_position]['epoch'] == epoch_equivalent:
            save(model, optimizer,scheduler, best_checkpointer_name, epoch_i)



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

#        total_train_all_correct = 0
#        total_train_average_correct = 0
#        total_train_all_head_correct = 0
#        total_train_average_head_correct = 0
#        total_train_all_reln_correct = 0
#        total_train_average_reln_correct = 0
#        total_train_all_tail_correct = 0
#        total_train_average_tail_correct = 0

        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2].to(device)
            #b_ground_truth_tokens = batch[3].to(device)

#            print(b_input_ids.size())
#            print(b_masks.size())
#            print(b_labels.size())
#            print(b_ground_truth_tokens.size())
#            print(b_labels)
#            print(b_input_ids[0])
#            print(tokenizer.decode(b_input_ids[0]))
#            raise ValueError
            model.zero_grad()

            outputs = model(  b_input_ids,
                              labels=b_labels,
                              attention_mask=b_masks,
                              token_type_ids=None
                            )

            loss = outputs[0]
            b_logits = outputs[1]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            #tokens = calculate_token_wise_accuracy(b_labels, b_logits) #scores,

#
#            average_correct, all_correct, \
#            average_head_correct, all_head_correct, \
#            average_reln_correct, all_reln_correct, \
#            average_tail_correct, all_tail_correct = scores

            total_train_ppl += 2**batch_loss

#            total_train_all_correct += all_correct
#            total_train_average_correct +=average_correct
#            total_train_all_head_correct += all_head_correct
#            total_train_average_head_correct += average_head_correct
#            total_train_all_reln_correct += all_reln_correct
#            total_train_average_reln_correct += average_reln_correct
#            total_train_all_tail_correct += all_tail_correct
#            total_train_average_tail_correct += average_tail_correct


            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs+starting_epoch))
                print(' Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}'.format(step, len(train_dataloader), batch_loss, elapsed))
#                print(' Average_correct: {} All_correct: {}'.format(total_train_average_correct/(step+1),total_train_all_correct/(step+1)))
#                print(' Average_head_correct: {} All_head_correct: {} '.format(total_train_average_head_correct/(step+1),total_train_all_head_correct/(step+1)))
#                print(' Average_reln_correct: {} All_reln_correct: {} '.format(total_train_average_reln_correct/(step+1),total_train_all_reln_correct/(step+1)))
#                print(' Average_tail_correct: {} All_tail_correct: {} '.format(total_train_average_tail_correct/(step+1),total_train_all_tail_correct/(step+1)))
                print(' ppl {} '.format(total_train_ppl/(step+1)))


                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()

            if (step+1) % int(eval_every * len(train_dataloader)) == 0 and step != 0:
                # Calculate the average loss over all of the batches.
                avg_train_loss = total_train_loss / (step+1)
                avg_train_ppl = total_train_ppl/ (step+1)
#                avg_train_all_correct = total_train_all_correct / (step+1)
#                avg_train_average_correct = total_train_average_correct / (step+1)
#                avg_train_all_head_correct = total_train_all_head_correct / (step+1)
#                avg_train_average_head_correct = total_train_average_head_correct / (step+1)
#                avg_train_all_reln_correct = total_train_all_reln_correct / (step+1)
#                avg_train_average_reln_correct = total_train_average_reln_correct / (step+1)
#                avg_train_all_tail_correct = total_train_all_tail_correct / (step+1)
#                avg_train_average_tail_correct = total_train_average_tail_correct / (step+1)
                # Measure how long this epoch took.
                training_time = format_time(time.time() - t0)

                eval_once(epoch_i + (step+1)/len(train_dataloader))

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

    else:
        eval_once(epoch_i + 1)
        break

    #eval_once(epoch_i + 1)

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
