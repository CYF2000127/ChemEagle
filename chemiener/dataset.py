import os
import cv2
import copy
import random
import json
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from transformers import BertTokenizerFast, AutoTokenizer, RobertaTokenizerFast

from .utils import get_class_to_index



class NERDataset(Dataset):
    def __init__(self, args, data_file, split='train'):
        super().__init__()
        self.args = args
        if data_file:
            data_path = os.path.join(args.data_path, data_file)
            with open(data_path) as f:
                self.data = json.load(f)
            self.name = os.path.basename(data_file).split('.')[0]
        self.split = split
        self.is_train = (split == 'train')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.roberta_checkpoint, cache_dir = self.args.cache_dir)#BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')
        self.class_to_index = get_class_to_index(self.args.corpus)
        self.index_to_class = {self.class_to_index[key]: key for key in self.class_to_index}

    #commment
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text_tokenized = self.tokenizer(self.data[str(idx)]['text'], truncation = True, max_length = self.args.max_seq_length)
        if len(text_tokenized['input_ids']) > 512: print(len(text_tokenized['input_ids']))
        text_tokenized_untruncated = self.tokenizer(self.data[str(idx)]['text']) 
        return text_tokenized, self.align_labels(text_tokenized, self.data[str(idx)]['entities'], len(self.data[str(idx)]['text'])), self.align_labels(text_tokenized_untruncated, self.data[str(idx)]['entities'], len(self.data[str(idx)]['text']))

    def align_labels(self, text_tokenized, entities, length):
        char_to_class = {}

        for entity in entities: 
            for span in entities[entity]["span"]:
                for i in range(span[0], span[1]):
                    char_to_class[i] = self.class_to_index[('B-' if i == span[0] else 'I-')+str(entities[entity]["type"])]

        for i in range(length):
            if i not in char_to_class:
                char_to_class[i] = 0
        
        classes = []
        for i in range(len(text_tokenized[0])):
            span = text_tokenized.token_to_chars(i)
            if span is not None:
                classes.append(char_to_class[span.start])
            else:
                classes.append(-100)

        return torch.LongTensor(classes)
    
    def make_html(word_tokens, predictions):
        
        toreturn = '''<!DOCTYPE html>
    <html>
    <head>
        <title>Named Entity Recognition Visualization</title>
        <style>
            .EXAMPLE_LABEL {
                color: red;
                text-decoration: underline red;
            }
            .REACTION_PRODUCT {
                color: orange;
                text-decoration: underline orange;
            }
            .STARTING_MATERIAL {
                color: gold;
                text-decoration: underline gold;
            }
            .REAGENT_CATALYST {
                color: green;
                text-decoration: underline green;
            }
            .SOLVENT {
                color: cyan;
                text-decoration: underline cyan;
            }
            .OTHER_COMPOUND {
                color: blue;
                text-decoration: underline blue;
            }
            .TIME {
                color: purple;
                text-decoration: underline purple;
            }
            .TEMPERATURE {
                color: magenta;
                text-decoration: underline magenta;
            }
            .YIELD_OTHER {
                color: palegreen;
                text-decoration: underline palegreen;
            }
            .YIELD_PERCENT {
                color: pink;
                text-decoration: underline pink;
            }
        </style>
    </head>
    <body>
        <p>'''
        last_label = None
        for idx, item in enumerate(word_tokens):
            decoded = self.tokenizer.decode(item, skip_special_tokens = True)
            if len(decoded)>0:
                if idx!=0 and decoded[0]!='#':
                    toreturn+=" "
                label = predictions[idx]
                if label == last_label:
                    
                    toreturn+=decoded if decoded[0]!="#" else decoded[2:]
                else:
                    if last_label is not None and last_label>0:
                        toreturn+="</u>"
                    if label >0:
                        toreturn+="<u class=\""
                        toreturn+=self.index_to_class[label]
                        toreturn+="\">"
                        toreturn+=decoded if decoded[0]!="#" else decoded[2:]
                    if label == 0:
                        toreturn+=decoded if decoded[0]!="#" else decoded[2:]
                if idx==len(word_tokens) and label>0:
                    toreturn+="</u>"
                last_label = label
        
        toreturn += '''    </p>
        </body>
        </html>'''
        return toreturn


def get_collate_fn():
    def collate(batch):
        


        sentences = []
        masks = []
        refs = []
  

        for ex in batch:
            sentences.append(torch.LongTensor(ex[0]['input_ids']))
            masks.append(torch.Tensor(ex[0]['attention_mask']))
            refs.append(ex[1])

        sentences = pad_sequence(sentences, batch_first = True, padding_value = 0) 
        masks = pad_sequence(masks, batch_first = True, padding_value = 0)
        refs = pad_sequence(refs, batch_first = True, padding_value = -100)
        return sentences, masks, refs

    return collate



