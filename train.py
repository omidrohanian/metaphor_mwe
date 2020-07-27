import torch
import gc
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm, trange
random_seed = 616

from evaluate import Evaluate

def train_test_loader(X, y, A, A_MWE, target_indices, k, batch_train, batch_test):
    """Generate k-fold splits given X, y, and the adjacency matrix A"""
    random_state = random_seed
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in X:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)

    kf = KFold(n_splits=k, random_state=random_state, shuffle=True)

    X = X.numpy()

    indices = np.array(target_indices)   # target token indexes
    A_MWE = np.array(A_MWE)

    for train_index, test_index in kf.split(X): 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        A_train, A_test = A[train_index], A[test_index]
        A_MWE_train, A_MWE_test = A_MWE[train_index], A_MWE[test_index]
        indices_train, indices_test = indices[train_index], indices[test_index]  # target token indexes

        train_masks, test_masks = attention_masks[train_index], attention_masks[test_index]

        train_indices = torch.tensor(train_index)
        test_indices = torch.tensor(test_index)    # these are actual indices which are going to be used for retrieving items after prediction

        # Convert to torch tensors
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)

        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)

        A_train = torch.tensor(A_train).long()
        A_test = torch.tensor(A_test).long()

        A_MWE_train = torch.tensor(A_MWE_train).long()
        A_MWE_test = torch.tensor(A_MWE_test).long()

        train_masks = torch.tensor(train_masks)
        test_masks = torch.tensor(test_masks)

        indices_train = torch.tensor(indices_train)
        indices_test = torch.tensor(indices_test)

        # Create an iterator with DataLoader
        train_data = TensorDataset(X_train, train_masks, A_train, A_MWE_train, y_train, indices_train, train_indices)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_train, drop_last=True)

        test_data = TensorDataset(X_test, test_masks, A_test, A_MWE_test, y_test, indices_test, test_indices)
        test_dataloader = DataLoader(test_data, sampler=None, batch_size=batch_test)

        yield train_dataloader, test_dataloader


def trainer(epochs, model, optimizer, scheduler, train_dataloader, test_dataloader, batch_train, batch_test, device):

    max_grad_norm = 1.0
    train_loss_set = []

    for e in trange(epochs, desc="Epoch"):

        while gc.collect() > 0:
            pass

        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # if e > 8:
        #     model.freeze_bert()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_adj, b_adj_mwe, b_labels, b_target_idx, _ = batch

            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            ### For BERT + GCN and MWE
            loss = model(b_input_ids.to(device), adj=b_adj, adj_mwe=b_adj_mwe ,attention_mask=b_input_mask.to(device), \
                        labels=b_labels, batch=batch_train, target_token_idx=b_target_idx.to(device))
        
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        all_preds = torch.FloatTensor()
        all_labels = torch.LongTensor()
        test_indices = torch.LongTensor()

        # Evaluate data for one epoch
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_adj, b_adj_mwe, b_labels, b_target_idx, test_idx = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                ### For BERT + GCN and MWE
                logits = model(b_input_ids.to(device), adj=b_adj, adj_mwe=b_adj_mwe, attention_mask=b_input_mask.to(device), \
                               batch=batch_test, target_token_idx=b_target_idx.to(device))

                # Move logits and labels to CPU
                logits = logits.detach().cpu()
                label_ids = b_labels.cpu()
                test_idx = test_idx.cpu()

                all_preds = torch.cat([all_preds, logits])
                all_labels = torch.cat([all_labels,label_ids])
                test_indices = torch.cat([test_indices, test_idx])

    scores = Evaluate(all_preds,all_labels)
    print('scores.accuracy()={}\nscores.precision_recall_fscore()={}'.format(scores.accuracy(),scores.precision_recall_fscore()))

    return scores, all_preds, all_labels, test_indices


