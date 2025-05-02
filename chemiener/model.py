import torch
from torch import nn


from transformers import BertForTokenClassification, RobertaForTokenClassification, AutoModelForTokenClassification


def build_model(args):
    if args.corpus == "chemu":
        return AutoModelForTokenClassification.from_pretrained(args.roberta_checkpoint, num_labels = 21, cache_dir = args.cache_dir, return_dict = False)
    elif args.corpus == "chemdner":
        return AutoModelForTokenClassification.from_pretrained(args.roberta_checkpoint, num_labels = 17, cache_dir = args.cache_dir, return_dict = False)
    elif args.corpus == "chemdner-mol":
        return AutoModelForTokenClassification.from_pretrained(args.roberta_checkpoint, num_labels = 3, cache_dir = args.cache_dir, return_dict = False)  
