import torch
import spacy
import numpy as np

def adjacency(sentences,max_len):
    """compute dependent-to-head adjacency matrices"""
    nlp = spacy.load("en_core_web_sm")
    A = []
    for sent in sentences:
        doc = nlp(sent)
        adj = np.zeros([max_len,max_len])
        for tok in doc:
            if not str(tok).isspace():
                if tok.i+1<max_len and tok.head.i+1<max_len:
                    adj[tok.i+1][tok.head.i+1] = 1
        A.append(adj)
    return A


def pad_or_truncate(input_ids, max_len):
        pad = lambda seq,max_len : seq[0:max_len] if len(seq) > max_len else seq + [0] * (max_len-len(seq))
        return torch.Tensor([pad(seq,max_len) for seq in input_ids]).long()

