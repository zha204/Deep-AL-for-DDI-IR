import numpy as np
import pandas as pd
import re
import os
import time
import random
from tqdm import tqdm
import dill
import spacy
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator, TabularDataset

from utils import binary_accuracy, calculate_metrics, epoch_time
from utils import train, evaluate, load_data, generate_external_sample, train_external, test_external, build_vocabulary, load_model
from model import FastText, CNN, RNN

def run_model(iteration, nlp, mode, model_name, SEED, batch_size, device, select_method, top_num, top_sim_num, n_epochs):
    print("loading data...")
    TEXT, LABEL, train_data, test_data, train_df, test_df, external_df = load_data(nlp, iteration, mode, model_name)
    TEXT, LABEL = build_vocabulary(TEXT, LABEL, train_data)
    
    train_data, val_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))
    
    train_iterator, valid_iterator, test_iterator = data.Iterator.splits(
        (train_data, val_data, test_data), 
        batch_size = batch_size,
        sort_key = lambda x: len(x.text),
        sort_within_batch = True,
        device = device)
    
    train_labels = [x.label for x in train_iterator.data()]
    labels_counter = Counter(train_labels)
    pos_weight = labels_counter['0']/labels_counter['1']
    
    pos_weight = torch.tensor(pos_weight)
    print('pos_weight', pos_weight)
    
    print("define the model...")
    model = load_model(model_name, TEXT)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = n_epochs
    best_valid_loss = float('inf')
    print("number of training samples:", len(train_iterator))
    print("number of validation samples:", len(valid_iterator))
    print("start training...")
    save_model_path = "../data/" + "iteration_" + str(iteration) + "/best_model_" + mode + ".pt"
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, model_name)
        valid_loss = evaluate(model, valid_iterator, criterion, model_name) 
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_model_path)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}')       

 
    print("predicting on external datasets...s")
    model.load_state_dict(torch.load(save_model_path))
    pred_y = []
    model.eval()
    with torch.no_grad():
        for i,row in tqdm(external_df.iterrows()):
            tensor = generate_external_sample(model_name, TEXT, row['text'], nlp)
            prediction = torch.sigmoid(model(tensor.to(device)))
            pred_y.append(prediction.item())
   
    print("generating uncertain samples...")
    pred_y = np.array(pred_y)
    if select_method == "range-46":
        uncertain_idx = np.where((pred_y >= 0.4) & (pred_y <= 0.6))[0]
        remain_idx = np.where((pred_y < 0.4) | (pred_y > 0.6))[0]
    elif select_method == "median-10":
        lower_45 = np.percentile(pred_y, 45)
        upper_55 = np.percentile(pred_y, 55)
        uncertain_idx = np.where((pred_y >= lower_45) & (pred_y <= upper_55))[0]
        remain_idx = np.where((pred_y < lower_45) | (pred_y > upper_55))[0]
    elif select_method == "random":
        temp_ids = list(range(external_df.shape[0]))
        np.random.shuffle(temp_ids)
        uncertain_idx = temp_ids[:top_num]
        remain_idx = temp_ids[top_num:]
    elif select_method == "range-top":
        order_idx = np.argsort(np.abs(pred_y - 0.5))
        uncertain_idx = order_idx[:top_num]
        remain_idx = order_idx[top_num:]
    
    print("number of uncertain samples:", len(uncertain_idx))
    print("the mean score of uncertain samples:", np.mean(pred_y[uncertain_idx]))
    
    uncertain_temp_df = external_df.iloc[uncertain_idx]
    uncertain_sim = uncertain_temp_df['sim'].values
    order_uncertain_sim = np.argsort(-uncertain_sim)
    uncertain_sim_idx = order_uncertain_sim[:top_sim_num]
    remain_sim_idx = order_uncertain_sim[top_sim_num:]
    remain_temp_df = uncertain_temp_df.iloc[remain_sim_idx]

    uncertain_df = uncertain_temp_df.iloc[uncertain_sim_idx]
    uncertain_df['iteration'] = iteration + 1
    new_external = pd.concat([external_df.iloc[remain_idx], remain_temp_df], ignore_index=True)

    save_text_path = "../data/" + "iteration_" + str(iteration) + "/Text_" + mode
    with open(save_text_path, "wb") as f:
            dill.dump(TEXT, f)
    
    return model, TEXT, uncertain_df, new_external