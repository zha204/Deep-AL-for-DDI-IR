import numpy as np
import pandas as pd
import time
import os
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext import datasets
from torchtext import data
from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from bisect import bisect_right, bisect_left
import matplotlib.pyplot as plt
from model import FastText, CNN, RNN, FastText_2

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

def generate_external_sample(model_name, TEXT, text, nlp):
    if model_name == "FastText" or model_name == "FastText_2":
        tokenized = generate_bigrams([tok.text for tok in nlp.tokenizer(text)])
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed)
        tensor = tensor.unsqueeze(1)
    elif model_name == "CNN":
        tokenized = [tok.text for tok in nlp.tokenizer(text)]
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed)
        tensor = tensor.unsqueeze(0)
    return tensor

def generate_initial_dataset(save_path, init_pos, init_unscr_neg, init_val):
    screened_corpus = pd.read_csv("../data/screened_corpus_df.csv")
    df_positive = screened_corpus[screened_corpus['label']==1]
    df_negative = screened_corpus[screened_corpus['label']==0]
    
    ids_pos = list(range(df_positive.shape[0]))
    np.random.shuffle(ids_pos)
    train_pos = df_positive.iloc[ids_pos[:init_pos]]
    
    ids_neg = list(range(df_negative.shape[0]))
    np.random.shuffle(ids_neg)
    train_scr_neg = df_negative.iloc[ids_neg[:init_pos]]
    
    train_df_scr = pd.concat([train_pos, train_scr_neg], ignore_index=True)
    train_df_scr['iteration'] = 0
    
    external_screened = pd.read_csv("../data/pool_screened_df.csv")
    
    df_unscreened_pool = pd.read_csv("../data/pool_unscreened_df.csv")
    ids = list(range(df_unscreened_pool.shape[0]))
    np.random.shuffle(ids)
    unscreened_neg = df_unscreened_pool.iloc[ids[:init_unscr_neg]]
    unscreened_neg['label'] = 0
    train_df_unscr = pd.concat([train_pos, unscreened_neg], ignore_index=True)
    train_df_unscr['iteration'] = 0
    
    external_unscreened = df_unscreened_pool.iloc[ids[init_unscr_neg:-init_val]]
    
    test_scr_pos = df_positive.iloc[ids_pos[-init_val:]]
    test_scr_neg = df_negative.iloc[ids_neg[-init_val:]]
    test_unscr_neg = df_unscreened_pool.iloc[ids[-init_val:]]
    test_unscr_neg.loc[test_unscr_neg['label']==-1,'label'] = 0
    test_df = pd.concat([test_scr_pos, test_scr_neg, test_unscr_neg], ignore_index=True)
    test_df['iteration'] = 0
    
    train_df_scr.to_csv(os.path.join(save_path,"train_screened_df.csv"), index=False)
    external_screened.to_csv(os.path.join(save_path,"external_screened_df.csv"), index=False)
    train_df_unscr.to_csv(os.path.join(save_path,"train_unscreened_df.csv"), index=False)
    external_unscreened.to_csv(os.path.join(save_path,"external_unscreened_df.csv"), index=False)
    test_df.to_csv(os.path.join(save_path,"test_df.csv"), index=False)

def assign_uncertain_samples(iteration, sharing_pos=0):
    pre_path = "../data/iteration_" + str(iteration - 1)
    uncertain_screen_df = pd.read_csv(pre_path + "/uncertain_screened.csv")
    train_screen_df = pd.read_csv(pre_path + "/train_screened_df.csv")
    uncertain_unscreen_df = pd.read_csv(pre_path + "/uncertain_unscreened.csv")
    train_unscreen_df = pd.read_csv(pre_path + "/train_unscreened_df.csv")
    test_df = pd.read_csv(pre_path + "/test_df.csv")
    
    uncertain_screen_pos = uncertain_screen_df[uncertain_screen_df['label'] == 1]
    uncertain_screen_neg = uncertain_screen_df[uncertain_screen_df['label'] == 0]
    uncertain_unscreen_pos = uncertain_unscreen_df[uncertain_unscreen_df['label'] == 1]
    uncertain_unscreen_neg = uncertain_unscreen_df[uncertain_unscreen_df['label'] == 0]
    print("positive samples in screened pool:", uncertain_screen_pos.shape[0])
    print("positive samples in unscreened pool:", uncertain_unscreen_pos.shape[0])
    
    def split_half(df):
        ids = list(range(df.shape[0]))
        np.random.shuffle(ids)
        df_first = df.iloc[ids[:int(0.5*df.shape[0])]]
        df_last = df.iloc[ids[int(0.5*df.shape[0]):]]
        return df_first, df_last
    
    uc_screen_pos_train, uc_screen_pos_test = split_half(uncertain_screen_pos)
    uc_screen_neg_train, uc_screen_neg_test = split_half(uncertain_screen_neg)
    uc_unscreen_pos_train, uc_unscreen_pos_test = split_half(uncertain_unscreen_pos)
    uc_unscreen_neg_train, uc_unscreen_neg_test = split_half(uncertain_unscreen_neg)
    
    if sharing_pos == 0:
        train_screen_next = pd.concat([train_screen_df, 
                                       uc_screen_pos_train, 
                                       uc_screen_neg_train], ignore_index=True).reset_index(drop=True)
        train_unscreen_next = pd.concat([train_unscreen_df, 
                                       uc_unscreen_pos_train, 
                                       uc_unscreen_neg_train], ignore_index=True).reset_index(drop=True)
    else:
        train_screen_next = pd.concat([train_screen_df, 
                                       uc_screen_pos_train, 
                                       uc_screen_neg_train,
                                       uc_unscreen_pos_train], ignore_index=True).reset_index(drop=True)
        train_unscreen_next = pd.concat([train_unscreen_df, 
                                       uc_unscreen_pos_train, 
                                       uc_unscreen_neg_train,
                                       uc_screen_pos_train], ignore_index=True).reset_index(drop=True)
    
    test_next = pd.concat([test_df, 
                           uc_screen_pos_test,
                           uc_screen_neg_test,
                           uc_unscreen_pos_test,
                           uc_unscreen_neg_test], ignore_index=True).reset_index(drop=True)
    cur_path = "../data/iteration_" + str(iteration)
    train_screen_next.to_csv(cur_path + "/train_screened_df.csv", index=False)
    train_unscreen_next.to_csv(cur_path + "/train_unscreened_df.csv", index=False)
    test_next.to_csv(cur_path + "/test_df.csv", index=False)
    

def load_data(nlp, iteration, mode="screened", model="FastText"):
    
    def tokenizer(text): 
        return [tok.text for tok in nlp.tokenizer(text)]

    print("MODEL:", model)
    if model == "FastText" or model == "FastText_2":
        TEXT = Field(sequential=True, 
                     tokenize=tokenizer, 
                     lower=True, 
                     preprocessing=generate_bigrams)
    
    LABEL = Field(sequential=False, 
                  unk_token=None, 
                  is_target=True,
                  dtype=torch.float)

    print("spliting datasets...")
    path = "../data/iteration_" + str(iteration)
    train, test = TabularDataset.splits(
            path=path, train='train_' + mode + '_df.csv', test='test_df.csv', format='csv', skip_header=True,
            fields=[('id',None), ('text', TEXT), ('label', LABEL)])
    
    train_df = pd.read_csv(path + '/train_' + mode + '_df.csv')
    test_df = pd.read_csv(path + "/test_df.csv")
    external_df = pd.read_csv(path + "/external_" + mode + "_df.csv")

    return TEXT, LABEL, train, test, train_df, test_df, external_df

def load_model(model_name, TEXT):
    if model_name == "FastText":
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 300
        OUTPUT_DIM = 1
        DROPOUT = 0.5
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
   
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    
    return model
    
def build_vocabulary(TEXT, LABEL, train):
    print("building glove vectors...")
    MAX_VOCAB_SIZE = 80000
    TEXT.build_vocab(train, 
                     max_size = MAX_VOCAB_SIZE,
                     vectors="glove.6B.300d",
                     unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train)
    print("number of vocab:", len(TEXT.vocab))
    
    return TEXT, LABEL

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def calculate_fmax(true_y, pred_y):
    pred_y = np.round(pred_y, 2)
    true_y = true_y.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (pred_y > threshold).astype(np.int32)
        tp = np.sum(predictions * true_y)
        fp = np.sum(predictions) - tp
        fn = np.sum(true_y) - tp
        sn = tp / (1.0 * np.sum(true_y))
        sp = np.sum((predictions ^ 1) * (true_y ^ 1))
        sp /= 1.0 * np.sum(true_y ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, r_max, p_max, t_max

def calculate_pre_recall(precision, recall, thres):
    precision = np.flip(precision)
    recall = np.flip(recall)
    idx = bisect_left(recall, thres)
    return precision[idx]

def calculate_metrics(true_y, pred_y):
    auc = roc_auc_score(true_y, pred_y)
    ap = average_precision_score(true_y, pred_y)
    f1 = f1_score(true_y, np.round(pred_y))
    precision, recall, thresholds = precision_recall_curve(true_y, pred_y)
    precision_99 = calculate_pre_recall(precision, recall,0.99)
    precision_95 = calculate_pre_recall(precision, recall,0.95)
    precision_90 = calculate_pre_recall(precision, recall,0.90)
    f_max, r_max, p_max, t_max = calculate_fmax(np.array(true_y), np.array(pred_y))
    report = classification_report(true_y, np.round(pred_y))
    
    return auc, ap, f1, report, precision_99, precision_95, precision_90, f_max, t_max

def plot_PRCurve(true_y, pred_y):
    precision, recall, _ = precision_recall_curve(true_y, pred_y)
    aupr = average_precision_score(true_y, pred_y)
    
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
    'Average precision score: AP={0:0.2f}'.format(aupr))
    plt.show()

def train(model, iterator, optimizer, criterion, model_name):  
    epoch_loss = 0
    epoch_acc = 0  
    model.train()
    
    for batch in iterator:      
        optimizer.zero_grad()
        if model_name == "RNN":
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
        else:
            predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)   
        acc = binary_accuracy(predictions, batch.label)   
        loss.backward()  
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_external(model, data_df, optimizer, criterion, model_name, TEXT, nlp, device):
    epoch_loss = 0
    epoch_acc = 0  
    model.train()
    
    for i, row in data_df.iterrows():      
        optimizer.zero_grad()
        text = generate_external_sample(model_name, TEXT, row['text'], nlp)
        text = text.to(device)
        if model_name == "RNN":
            text_lengths = text.shape[0]
            predictions = model(text, text_lengths).squeeze(1)
        else:
            predictions = model(text).squeeze(1)
        true_y = torch.tensor(row['label'], dtype=torch.float).unsqueeze(0).to(device)
        loss = criterion(predictions, true_y)   
        acc = binary_accuracy(predictions, true_y)   
        loss.backward()  
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / data_df.shape[0], epoch_acc / data_df.shape[0]

def test_external(model, data_df, criterion, model_name, TEXT, nlp, device):
    epoch_loss = 0
    pred_y_list = []
    true_y_list = data_df['label'].values.tolist()
    model.eval()
    with torch.no_grad():
        for i,row in data_df.iterrows():
            text = generate_external_sample(model_name, TEXT, row['text'], nlp)
            text = text.to(device)
            if model_name == "RNN":
                text_lengths = text.shape[0]
                predictions = model(text, text_lengths).squeeze(1)
            else:
                predictions = model(text).squeeze(1)
            true_y = torch.tensor(row['label'], dtype=torch.float).unsqueeze(0).to(device)
            loss = criterion(predictions, true_y)   
            epoch_loss += loss.item()
            predictions = torch.sigmoid(predictions)
            pred_y_list.append(predictions.cpu().item())
        auc, ap, f1, report, precision_99, precision_95, precision_90, f_max, t_max = calculate_metrics(true_y_list, pred_y_list)
    return epoch_loss / len(data_df), auc, ap, f1, report, precision_99, precision_95, precision_90, f_max, t_max, true_y_list, pred_y_list
        

def evaluate(model, iterator, criterion, model_name):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    true_y = []
    pred_y = []
    
    with torch.no_grad():
        for batch in iterator:
            if model_name == "RNN":
                text, text_lengths = batch.text
                predictions = model(text, text_lengths).squeeze(1)
            else:
                predictions = model(batch.text).squeeze(1) 
            loss = criterion(predictions, batch.label) 
            predictions = torch.sigmoid(predictions)
            true_y.extend(batch.label.cpu().numpy().ravel().tolist())
            pred_y.extend(predictions.cpu().numpy().ravel().tolist())

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs