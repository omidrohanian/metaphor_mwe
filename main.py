import sys
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from evaluate import Evaluate
from tqdm import tqdm_notebook

import re, spacy, copy, random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, BertModel
from layers.GCN import *
from tqdm import tqdm, trange
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import time
import gc

from mwe.myMWEProcess import *
from train import *
from models import *
from utils import *

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_file_path = sys.argv[1]

    with open(config_file_path) as f:
        config = json.load(f)

    file_dir = config["file_dir"]
    mwe_dir = config["mwe_dir"]

    BATCH_TRAIN = config["batch_train"]
    BATCH_TEST = config["batch_test"]

    K = config["K"]
    EPOCHS = config["epochs"]
    dropout = config["dropout"]

    num_total_steps = config["num_total_steps"]
    num_warmup_steps = config["num_warmup_steps"]
        
    max_grad_norm = 1.0

    df = pd.read_csv(file_dir, header=0, sep=',')
    # Create sentence and label lists
    sentences = df.sentence.values

    MAX_LEN = max([len(sent.split()) for sent in sentences]) + 2
    print('MAX_LEN =',MAX_LEN)

    MAX_LEN = config["max_len"]

    A = np.array(adjacency(sentences=sentences,max_len=MAX_LEN))

    with open(mwe_dir) as f:
            A_MWE = mwe_adjacency(f, file_dir, MAX_LEN-2)

    nlp = spacy.load("en_core_web_sm")
    # tokenize sentences
    # the same tokenizer that is used to get adjacency matrices 
    tokenized_texts = []
    for sent in sentences:
            tokenized_sent = []
            doc = nlp(sent)
            for token in doc:
                if not token.text.isspace():
                    tokenized_sent.append(token.text.lower())
            tokenized_texts.append(tokenized_sent)

    # add special tokens at the beginning and end of each sentence
    for sent in tokenized_texts:
            sent.insert(0,'[CLS]')
            sent.insert(len(sent),'[SEP]')

    print('len(sentences)={}'.format(len(sentences)))

    labels = df['label'].values

    target_token_idices = df['verb_idx'].values

    print('max_len of tokenized texts:',max([len(sent) for sent in tokenized_texts]))

    print ("Tokenize the first sentence:")
    print (tokenized_texts[0])

    # construct the vocabulary 
    vocab = list(set([w for sent in tokenized_texts for w in sent]))
    # index the input words 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    input_ids = pad_or_truncate(input_ids,MAX_LEN)    

    bert_config = BertConfig(vocab_size_or_config_json_file=len(vocab))

    heads = config["heads"]
    heads_mwe = config["heads_mwe"]


    all_test_indices = []
    all_predictions = []
    all_folds_labels = []
    recorded_results_per_fold = []
    splits = train_test_loader(input_ids, labels, A, A_MWE, target_token_idices, K, BATCH_TRAIN, BATCH_TEST)

    for i, (train_dataloader, test_dataloader) in enumerate(splits):
        model = BertWithGCNAndMWE(MAX_LEN, bert_config, heads, heads_mwe, dropout)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                     num_training_steps=num_total_steps)

        print('fold number {}:'.format(i+1))

        scores, all_preds, all_labels, test_indices = trainer(EPOCHS, model, optimizer, scheduler, 
                train_dataloader, test_dataloader, BATCH_TRAIN, BATCH_TEST, device)
        recorded_results_per_fold.append((scores.accuracy(),)+scores.precision_recall_fscore())

        all_test_indices.append(test_indices)
        all_predictions.append(all_preds)
        all_folds_labels.append(all_labels)

    print('K-fold cross-validation results:')
    print("Accuracy: {}".format(sum([i for i,j,k,l in recorded_results_per_fold])/K))
    print("Precision: {}".format(sum([j for i,j,k,l in recorded_results_per_fold])/K))
    print("Recall: {}".format(sum([k for i,j,k,l in recorded_results_per_fold])/K))
    print("F-score: {}".format(sum([l for i,j,k,l in recorded_results_per_fold])/K))

    # sanity checks
    print('####')
    print('recorded_results_per_fold=',recorded_results_per_fold)
    print('len(set(recorded_results_per_fold))=',len(set(recorded_results_per_fold)))
