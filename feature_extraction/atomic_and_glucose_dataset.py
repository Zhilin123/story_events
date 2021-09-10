#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset, random_split
from tqdm import trange
import torch
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from ast import literal_eval
import random
import collections
    
class AtomicDataset(Dataset):
    
    def __init__(self, filename,  tokenizer, max_length=128, debug_mode=False, data_subset="trn", add_negative_samples=False, structured=False, reverse=False):
        # structured isnt used for AtomicDataset
        
        assert data_subset in ["trn", "tst", "dev"]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.data_subset = data_subset
        
        df = pd.read_csv(filename)
        
        self.data_category = list(df["split"])
        
        event = [str(i) for i in list(df["event"])]
        event = self.filter_using_data_subset(event)
        
        prefix = list(df['prefix'])
        prefix = self.filter_using_data_subset(prefix)
        
        allowed_labels = ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"]
        reverse_labels = [i+'-r' for i in allowed_labels]
        allowed_labels += reverse_labels
        allowed_labels += ["xNone"]
        
        label_groups = [
                    ['xWant', 'oWant', 'xNeed', 'xIntent'],
                    ['xReact', 'oReact', 'xAttr'],
                    ['xEffect', 'oEffect']
                ]
        
        label_to_label_group = {}
        for label_group in label_groups:
            for label in label_group:
                label_to_label_group[label] = label_group
        
        
        self.label_to_key = {allowed_labels[i]: i for i in range(len(allowed_labels))}
        
        #label_of_interest = "xEffect"
        
        label_to_tail_nodes = {
                    allowed_label: self.filter_using_data_subset(
                                [[j for j in literal_eval(i) if j != "none"] for i in list(df[allowed_label])]
                            ) 
                        for allowed_label in allowed_labels[:9]
                }
        
        if debug_mode:
            event = event[:5]
        
        self.element_to_tokenized = {}
        
        positive_exs, negative_exs = 0, 0
        
        for i in trange(len(event)):
            for allowed_label in allowed_labels[:9]:
                #other_labels = [i for i in allowed_labels if i not in ["xNone", allowed_label]]
                for j in range(len(label_to_tail_nodes[allowed_label][i])):
                    self.add_event(event[i], 
                                   #self.transform_atomic_tailnode(label_to_tail_nodes[allowed_label][i][j], allowed_label), 
                                   label_to_tail_nodes[allowed_label][i][j],
                                   allowed_label)
                    
                    positive_exs += 1
                    
                    if reverse:
                        self.add_event(label_to_tail_nodes[allowed_label][i][j],
                                       event[i], 
                                       allowed_label+'-r')
                    
                        positive_exs += 1
                    
                    if add_negative_samples and i % 9 == 0:
                        #negative sampling of tail nodes associated with that event
                        random_label = random.choice(label_to_label_group[allowed_label])
                        random_i = random.randint(0,len(event)-1)
                        
                        counter = 0
                        while not label_to_tail_nodes[random_label][random_i] and prefix[i] == prefix[random_i] and counter < 10:
                            random_label = random.choice(label_to_label_group[allowed_label])
                            random_i = random.randint(0,len(event)-1)
                            counter += 1
                            
                        if (not label_to_tail_nodes[random_label][random_i]) or prefix[i] == prefix[random_i]:
                            continue
                        
                        random_tail_node = random.choice(label_to_tail_nodes[random_label][random_i])
                        
                        self.add_event(event[i], random_tail_node, "xNone")
                        
                        
                        negative_exs += 1
                        
                        if reverse:
                            self.add_event(random_tail_node, event[i], "xNone")
                            negative_exs += 1
                            
        print("Positive ex: {}, Negative ex: {}".format(positive_exs, negative_exs))
        del self.element_to_tokenized
        
        '''
        self.unique_masks = list(set(self.attn_masks))
        self.unique_mask_to_pos = {self.unique_masks[i]:i for i in range(len(self.unique_masks))}
        self.attn_mask_pos = [self.unique_mask_to_pos[i] for i in self.attn_masks]
        
        del self.attn_masks 
        del self.unique_mask_to_pos
        '''
    
    def add_event(self, head_node, tail_node, allowed_label):
        
        ground_elements = [head_node, #"[CLS]",
                           "[RELN]", # this is to allow for subsequent experiments with "xReact"
                           "[SEP]", tail_node, 
                           "[PAD]"]
        
        
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

        label = self.label_to_key[allowed_label]
        
        self.labels.append(label)
        self.input_ids.append(ground_input_ids)
        self.attn_masks.append(ground_attn_masks)
        
    def transform_atomic_tailnode(self, tailnode, attr):
        tailnode = tailnode.replace("persony", "PersonY")\
                           .replace("person y", "PersonY")\
                           .replace("Person Y", "PersonY")\
                           .replace("personx", "PersonX")\
                           .replace("person x", "PersonX")\
                           .replace("Person X", "PersonX")\
        
        if tailnode.split()[0].lower() in ['she', 'he', 'they', 'we', 'personx', 'persony', 'to']:
            tailnode = ' '.join(tailnode.split()[1:])
        
        if tailnode:
            tailnode = tailnode[0].lower() + tailnode[1:]
                 
        
        
        
        if attr in ["oReact", "xReact"]:
            verb = "feels "
        elif attr == "xAttr:":
            verb = "is "
        else:
            verb = ""
        
        subj = "PersonY " if attr[0] == "o" else "PersonX "
        
        tailnode = subj + verb + tailnode
        
        # changing the subject since for 
        # all parsed event the subject is always PersonX
        tailnode = tailnode.replace("PersonY", "PersonZ")\
                           .replace("PersonX", "PersonY")\
                           .replace("PersonZ", "PersonX")
        return tailnode

    def filter_using_data_subset(self, original_data):
        return [original_data[i] for i in range(len(original_data)) if self.data_category[i] == self.data_subset]
    
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, item):
        return self.input_ids[item], self.attn_masks[item], self.labels[item] #self.unique_masks[self.attn_mask_pos[item]], 
    
    def get_uniques_stable(self, one_list):
        dict_repr = {i:0 for i in one_list}
        return [i for i in dict_repr]
        

class GlucoseDataset(AtomicDataset):
    
    def __init__(self, filename,  tokenizer, max_length=128, debug_mode=False, data_subset="trn", add_negative_samples=False, structured=False, reverse=False):
        
        assert data_subset in ["trn", "tst", "dev"]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.data_subset = data_subset
        
        df = pd.read_csv(filename)
        
    
        self.data_category = self.get_data_category_splits(df)
        
        #event = [str(i) for i in list(df["event"])]
        #event = self.filter_using_data_subset(event)
        #allowed_labels = ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant", "xNone"]
        #
        
        column_ending_with = "specificStructured" if structured else "specificNL"
        
        allowed_labels = ["{}_{}".format(i,column_ending_with) for i in range(1,11)] + ["xNone"]
        
        self.label_to_key = {allowed_labels[i]: i for i in range(len(allowed_labels))}
        
        #label_of_interest = "xEffect"
        
        label_to_tail_nodes = {
                    allowed_label: self.filter_using_data_subset(
                                list(df[allowed_label])
                            ) 
                        for allowed_label in allowed_labels[:-1]
                }
        
        if debug_mode:
            for allowed_label in allowed_labels[:-1]:
                label_to_tail_nodes[allowed_label] = label_to_tail_nodes[allowed_label][:5]
            
        
        self.element_to_tokenized = {}
        
        for i in trange(len(label_to_tail_nodes[allowed_labels[0]])):
            for allowed_label in allowed_labels[:-1]:
                other_labels = ["{}_{}".format(i,column_ending_with) for i in range(1,6)] if int(allowed_label.split("_")[0]) > 5 else ["{}_{}".format(i,column_ending_with) for i in range(6,11)]
                
                if label_to_tail_nodes[allowed_label][i] == "escaped":
                    continue
                
                cause, _, effect = label_to_tail_nodes[allowed_label][i].split(">")
                if structured:
                    cause = GlucoseDataset.convert_glucose_annotation_to_str(cause)
                    effect = GlucoseDataset.convert_glucose_annotation_to_str(effect)
                
                
                self.add_event(cause, effect, allowed_label)
                
                if add_negative_samples:
                    #negative sampling of tail nodes associated with that event
                    random_label = random.choice(other_labels)
                    counter = 0
                    
                    while label_to_tail_nodes[random_label][i] != "escaped" and counter < 10:
                        random_label = random.choice(other_labels)
                        counter += 1
                        
                    if label_to_tail_nodes[random_label][i] != "escaped":
                        random_cause, _, random_effect = label_to_tail_nodes[random_label][i].split(">")
                        if structured:
                            random_cause = GlucoseDataset.convert_glucose_annotation_to_str(random_cause)
                            random_effect = GlucoseDataset.convert_glucose_annotation_to_str(random_effect)
                   
                        # for glucose the first 5 effects are mostly the same
                        # the last 5 causes are mostly the same
                        # we cannot change the effect with another random effect 
                        # because that random effect is also associated with the same cause and another label
                        if int(allowed_label.split("_")[0]) < 6:
                            self.add_event(random_effect, effect, "xNone")
                        else:
                            self.add_event(cause, random_cause, "xNone")
                                
        del self.element_to_tokenized
    
    def get_data_category_splits(self, df):
        one_unit = len(df) // 10
        trn_indices, tst_indices, dev_indices = random_split(range(len(df)), [one_unit*8+len(df)%10,one_unit,one_unit], generator=torch.Generator().manual_seed(42))
        
        data_category = [0] * len(df)
        
        for trn_index in trn_indices:
            data_category[trn_index] = "trn"
        for tst_index in tst_indices:
            data_category[tst_index] = "tst"
        for dev_index in dev_indices:
            data_category[dev_index] = "dev"
        return data_category
    
    @staticmethod
    def convert_glucose_annotation_to_dict(annotation):
        phrases = []
        labels = []
        val = ""
        
        for i in range(len(annotation)):
            if annotation[i] == "{":
                val = ""
            elif annotation[i] == "}":
                phrases.append(val)
            elif annotation[i] == "[":
                val = ""
            elif annotation[i] == "]":
                labels.append(val)
            else:
                val += annotation[i]
            
        ground_label_to_my_label = {
                    "subject": "SUBJ",
                    "verb": "VERB",
                    "preposition1": "PRP1",
                    "object1": "OBJ1",
                    "preposition": "PRP1",
                    "object": "OBJ1",
                    "preposition2": "PRP2",
                    "object2": "OBJ2",
                }
        
        my_label_to_phrase = collections.defaultdict(str)
        
        for i in range(len(phrases)):
            try:
                my_label_to_phrase[ground_label_to_my_label[labels[i]]] = phrases[i].lower().strip()
            except:
                print(labels[i])
        
        return my_label_to_phrase
    
    @staticmethod
    def convert_defaultdict_to_str(one_defaultdict):
        fields = ['SUBJ', 'VERB', 'PRP1', 'OBJ1', 'PRP2', 'OBJ2']
        field_value = [('['+field+']', one_defaultdict[field]) for field in fields]
        str_repr = ' '.join([i[0] + ' ' + i[1] for i in field_value])
        return str_repr
    
    @staticmethod
    def convert_glucose_annotation_to_str(annotation):
        return GlucoseDataset.convert_defaultdict_to_str(
                GlucoseDataset.convert_glucose_annotation_to_dict(annotation)
                )
        
class HippoAtomicDataset(AtomicDataset):
    
    def __init__(self, filename,  tokenizer, max_length=128, debug_mode=False, data_subset="recalled", add_negative_samples=False, structured=False, reverse=False, paragraph_level=False):
        
        assert data_subset in ["recalled", "imagined", "retold", "all"]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.data_subset = data_subset
        
        df = pd.read_csv(filename)
        
        self.element_to_tokenized = {}
        self.data_category = list(df['input_type'])
        
        allowed_labels = self.get_allowed_labels(structured)
        
        self.label_to_key = {allowed_labels[i]: i for i in range(len(allowed_labels))}
        
        
        if not structured: 
            paragraphs = list(df['paragraph'])
            if data_subset != "all":
                paragraphs = self.filter_using_data_subset(paragraphs)
            
            paragraphs = self.get_uniques_stable(paragraphs)
            paragraphs = [literal_eval(i) for i in paragraphs]
            
            for paragraph in paragraphs:
                for i in range(1, len(paragraph)):
                    start = 0 if paragraph_level else i-1
                    for j in range(start, i):
                        current_sentence = paragraph[j]
                        next_sentence = paragraph[i]
                        if not reverse:
                            self.add_event(current_sentence, next_sentence, allowed_labels[0])
                        else:
                            ## add the reverse event instead
                            self.add_event(next_sentence, current_sentence, allowed_labels[0])
        else:
            paragraphs = list(df['paragraph'])
            paragraphs = self.filter_using_data_subset(paragraphs)
            sentences = list(df['sentence'])
            sentences = self.filter_using_data_subset(sentences)
            structured_column = self.get_structured_column()
            events_data = self.filter_using_data_subset(list(df[structured_column]))
            events = [literal_eval(i) for i in events_data]
            for i in range(1, len(paragraphs)):
                start = max(i-4, 0) if paragraph_level else i-1
                for j in range(start, i):
                    if paragraphs[j] == paragraphs[i]:
                        current_events = events[j]
                        next_events = events[i]
                        # next_event in natural language
                        #next_events = [sentences[i+1]]
                        for current_event in current_events:
                            for next_event in next_events:
                                if not reverse:
                                    self.add_event(current_event, next_event, allowed_labels[0])
                                else:
                                    #add the reverse event
                                    self.add_event(next_event, current_event, allowed_labels[0])
                            
        del self.element_to_tokenized

    def get_allowed_labels(self, structured):
        return ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant", "xNone"]
    
    def get_structured_column(self):
        return 'atomic_structured'
    
class HippoGlucoseDataset(HippoAtomicDataset):
    
    def get_allowed_labels(self, structured):
        column_ending_with = "specificStructured" if structured else "specificNL"
        allowed_labels = ["{}_{}".format(i,column_ending_with) for i in range(1,11)] + ["xNone"]
        return allowed_labels
    
    def get_structured_column(self):
        return 'glucose_structured'


class StoryClozeAtomicDataset(HippoAtomicDataset):
    
    def __init__(self, filename,  tokenizer, max_length=128, debug_mode=False, data_subset="all", add_negative_samples=False, structured=False, reverse=False, paragraph_level=False):
        
        assert data_subset in ["recalled", "imagined", "retold", "all"]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.data_subset = data_subset
        
        df = pd.read_csv(filename)
        
        self.element_to_tokenized = {}
        #self.data_category = list(df['input_type'])
        
        allowed_labels = self.get_allowed_labels(structured)
        
        self.label_to_key = {allowed_labels[i]: i for i in range(len(allowed_labels))}
        
        first_four_sentences = [list(df["InputSentence{}".format(i)]) for i in range(1,5)]
        
        first_four_sentences_combined = [
                    [first_four_sentences[0][i],
                      first_four_sentences[1][i],
                      first_four_sentences[2][i],
                      first_four_sentences[3][i]] for i in range(len(first_four_sentences[0]))
                ]
        
        
        option_1 = list(df["RandomFifthSentenceQuiz1"])
        option_2 = list(df["RandomFifthSentenceQuiz2"])
        
        paragraphs_1 = [first_four_sentences_combined[i]+ [option_1[i]] for i in range(len(first_four_sentences_combined))]
        paragraphs_2 = [first_four_sentences_combined[i]+ [option_2[i]] for i in range(len(first_four_sentences_combined))]
        
        paragraphs = paragraphs_1 + paragraphs_2
        if not structured: 
            
            for paragraph in paragraphs:
                for i in range(1, len(paragraph)):
                    start = 0 if paragraph_level else i-1
                    for j in range(start, i):
                        current_sentence = paragraph[j]
                        next_sentence = paragraph[i]
                        if not reverse:
                            self.add_event(current_sentence, next_sentence, allowed_labels[0])
                        else:
                            ## add the reverse event instead
                            self.add_event(next_sentence, current_sentence, allowed_labels[0])
                            
class StoryClozeGlucoseDataset(StoryClozeAtomicDataset):
    
    def get_allowed_labels(self, structured):
        column_ending_with = "specificStructured" if structured else "specificNL"
        allowed_labels = ["{}_{}".format(i,column_ending_with) for i in range(1,11)] + ["xNone"]
        return allowed_labels
    
    def get_structured_column(self):
        return 'glucose_structured'

                            
if __name__ == "__main__":
    from transformers import RobertaTokenizer
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    filename = "../event2mind-internal-master/code/parsingcode/v4_atomic_all_agg.csv"
    train_dataset = AtomicDataset(
                filename,  tokenizer, max_length=128, debug_mode=True, data_subset="trn",reverse=True
            )
    
    print(train_dataset.labels)
    filename = "../GLUCOSE_training_data_final.csv"
    train_dataset = GlucoseDataset(
                filename,  tokenizer, max_length=128, debug_mode=True, data_subset="trn", add_negative_samples=True, structured=True
            )
    print(train_dataset.labels)

 