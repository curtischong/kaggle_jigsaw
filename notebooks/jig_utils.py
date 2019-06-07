import random
import os
import numpy as np
import pickle
import time

import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

# use to save the model metadata
import json
import datetime

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_embeddings(path):
    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr
    
def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

class SequenceBucketCollator():
    def __init__(self, choose_length, maxlen, sequence_index, length_index, label_index=None):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.label_index = label_index
        self.maxlen = maxlen
        
    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]
        
        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]
        
        length = self.choose_length(lengths)
        mask = torch.arange(start=self.maxlen, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]
        
        batch[self.sequence_index] = padded_sequences
        
        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]
    
        return batch
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_weights_dir(model_dir):
    try:
        os.mkdir(model_dir)
    except:
        pass

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

"""def train_model(model, train, test, loss_fn, output_dim, lr=0.001,
                batch_size=512, n_epochs=4, n_epochs_embed=2,
                enable_checkpoint_ensemble=True, filepath):
    # not sure if we have to redefine this after every iteration
    train_collator = SequenceBucketCollator(lambda lengths: lengths.max(), 
                                        sequence_index=0, 
                                        length_index=1, 
                                        label_index=2)
    
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=train_collator)
    val_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=train_collator)
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    best_loss = 1
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        scheduler.step()
        
        model.train() #set model to train mode
        avg_loss = 0.
        
        for data in tqdm(train_loader, disable=False):
            
            #training loop
            x_batch = data[:-1]

            first = x_batch[0][0]
            second = x_batch[0][1]
            
            y_batch = data[-1]

            y_pred = model(first, second)  #feed data into model          
            loss = loss_fn(y_pred, y_batch)
            
            #calculate error and adjust model params
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        
        #Check if loss is better than current best loss, if so, save the model
        is_best = (avg_loss < best_loss)
        if is_best:
            print ("=> Saving a new best")
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, filepath)  # save checkpoint
        else:
            print ("=> Model Accuracy did not improve")
            
        
        model.eval() #set model to eval mode for test data
        test_preds = np.zeros((len(test), output_dim))
        
    
        for i, x_batch in enumerate(val_loader):
            #print("X_Batch: ", x_batch)
            data_param = x_batch[0][0]
            lengths_param = x_batch[0][1]
            y_pred = sigmoid(model(data_param, lengths_param).detach().cpu().numpy()) #feed data into model

            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred #get test predictions
        print("freeing gpu ram")
        del x_batch
        torch.cuda.empty_cache()
        
        #test_preds has the predictions for the entire test set now
        all_test_preds.append(test_preds) #append predictions to the record of all past predictions
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
              epoch + 1, n_epochs, avg_loss, elapsed_time))
    return model"""

def predict(model, test, output_dim, batch_size=512, pred_type="val"):
    #checkpoint = torch.load(filepath)
    #model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if pred_type == "test":
        test_collator = SequenceBucketCollator(lambda lengths: lengths.max(), sequence_index=0, length_index=1)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=test_collator)

        model.eval() #set model to eval mode for test data
        test_preds = np.zeros((len(test), output_dim))

        for i, x_batch in enumerate(test_loader):
            #print(x_batch[0])
            data_param = x_batch[0]
            lengths_param = x_batch[1]
            y_pred = sigmoid(model(data_param, lengths_param).detach().cpu().numpy()) #feed data into model
            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred #get test predictions

        return test_preds
    else:
        test_collator = SequenceBucketCollator(lambda lengths: lengths.max(), sequence_index=0, length_index=1, label_index=2)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=test_collator)

        model.eval() #set model to eval mode for test data
        test_preds = np.zeros((len(test), output_dim))

        for i, x_batch in enumerate(test_loader):
            #print(x_batch)
            data_param = x_batch[0][0]
            lengths_param = x_batch[0][1]
            y_pred = sigmoid(model(data_param, lengths_param).detach().cpu().numpy()) #feed data into model
            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred #get test predictions

        return test_preds
    

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        #call the forward method in Dropout2d (super function specifies the subclass and instance)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
class JigsawEvaluator:
    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):
        self.y = y_true
        self.y_i = y_identity
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight

    @staticmethod
    def _compute_auc(y_true, y_pred):
        #print("Here: ", y_true)
        #print(y_pred)
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        #print(self.y)
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def compute_bias_metrics_for_model(self, y_pred):
        #print(y_pred)
        records = np.zeros((3, self.n_subgroups))
        for i in range(self.n_subgroups):
            #print(y_pred)
            records[0, i] = self._compute_subgroup_auc(i, y_pred)
            records[1, i] = self._compute_bpsn_auc(i, y_pred)
            records[2, i] = self._compute_bnsp_auc(i, y_pred)
        return records

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        bias_score = np.average([
            self._power_mean(bias_metrics[0]),
            self._power_mean(bias_metrics[1]),
            self._power_mean(bias_metrics[2])
        ])
        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)
        bias_score = (1 - self.overall_model_weight) * bias_score
        return overall_score + bias_score
    
def save_model_stats(model_notes,
                     num_splits,
                     dir_name,
                     num_models,
                     notebook_start_time,
                     final_val):
    model_data = {
        "model_notes": model_notes,
        "num_folds": num_splits,
        "models_per_fold": num_models,
        "time_ran": str(datetime.datetime.now()),
        "run_time": time.time() - notebook_start_time,
        "final_val": final_val
    }
    with open(dir_name + "info.json", 'w') as outfile:
        json.dump(model_data, outfile)
        print("saved model to %s" % dir_name)
        print(model_data)
        return
    print("Failed to save model")