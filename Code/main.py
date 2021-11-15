import numpy as np
import pandas as pd
import re
import os,sys
from shutil import copyfile
import time
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import spacy
import argparse
import json
import dill
from collections import defaultdict

from run_model import run_model
from utils import test_external, load_model, assign_uncertain_samples, generate_initial_dataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='FastText')
parser.add_argument('--batch_size', type=int, default=4) 
parser.add_argument('--max_iter', type=int, default=5) 
parser.add_argument('--starting_iter', type=int, default=0)
parser.add_argument('--evaluation', type=int, default=0)
parser.add_argument('--no_train', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=200) 
parser.add_argument('--select_method', type=str, default="range-top")
parser.add_argument('--top_num', type=int, default=50)  
parser.add_argument('--top_sim_num', type=int, default=20) 
parser.add_argument('--init_pos', type=int, default=50)  
parser.add_argument('--init_unscr_neg', type=int, default=50) 
parser.add_argument('--init_val', type=int, default=50) 
parser.add_argument('--sharing_pos', type=int, default=0)
parser.add_argument('--seed', type=int, default=2)
args = parser.parse_args()

print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)
print("loading spacy core...")
nlp = spacy.load("en_core_web_lg")

np.random.seed(args.seed)
torch.manual_seed(args.seed)


if not args.no_train:
    print("Training...")
    for i in range(args.starting_iter, args.starting_iter+1):
        print("#################################")
        print("iteration", i)
        print("#################################\n")
        
        if i > 0:
            print("assigning uncertain samples...")
            assign_uncertain_samples(i, args.sharing_pos)
        elif i == 0:
            path = "../data/iteration_0"
            if not os.path.exists(path):
                os.makedirs(path)
            generate_initial_dataset(path, args.init_pos, args.init_unscr_neg, args.init_val)

      
        print("********* Screened model *********\n")
        model_screen, TEXT_screen, uncertain_screen, external_next_screen = run_model(i, nlp, "screened", args.model, args.seed, args.batch_size, 
                                                                                      device, args.select_method, args.top_num, args.top_sim_num, args.n_epochs)

        
        print("\n********* Unscreened model *********\n")
        model_unscreen, TEXT_unscreen, uncertain_unscreen, external_next_unscreen = run_model(i, nlp, "unscreened", args.model, args.seed, args.batch_size, 
                                                                                              device, args.select_method, args.top_num,args.top_sim_num, args.n_epochs)

        print("saving data for next iteration...")
        next_path = "../data/iteration_" + str(i+1)
        if not os.path.exists(next_path):
            os.makedirs(next_path)
        external_next_screen.to_csv(next_path+"/external_screened_df.csv", index=False)
        external_next_unscreen.to_csv(next_path+"/external_unscreened_df.csv", index=False)
        
        cur_path = "../data/iteration_" + str(i)
        uncertain_screen.to_csv(cur_path+"/uncertain_screened.csv", index=False)
        uncertain_unscreen.to_csv(cur_path+"/uncertain_unscreened.csv", index=False)
        

if args.evaluation:
    print("evaluation...")
    models_screened = []
    models_unscreened = []
    TEXTs_screened = []
    TEXTs_unscreened = []
    
    print("loading models...")
    for i in range(args.max_iter):
        input_path = "../data/iteration_" + str(i)
        with open(input_path + "/Text_screened", "rb") as f:
            TEXT_screen = dill.load(f)
        with open(input_path + "/Text_unscreened", "rb") as f:
            TEXT_unscreen = dill.load(f)
            
        model_screen = load_model(args.model, TEXT_screen).to(device)
        model_screen.load_state_dict(torch.load(input_path + "/best_model_screened.pt"))
        model_unscreen = load_model(args.model, TEXT_unscreen).to(device)
        model_unscreen.load_state_dict(torch.load(input_path + "/best_model_unscreened.pt"))
        
        models_screened.append(model_screen)
        models_unscreened.append(model_unscreen)
        TEXTs_screened.append(TEXT_screen)
        TEXTs_unscreened.append(TEXT_unscreen)
    
    print("loading the final test set...")
    test_df = pd.read_csv("../data/iteration_" + str(args.max_iter-1) + "/test_df.csv")
    test_df_screened = test_df.loc[(test_df['source'] == "screened")]
    test_df_unscreened = test_df.loc[(test_df['label'] == 1) | (test_df['source'] == "unscreened")]
    
    print("Number of test samples for screened model:", test_df_screened.shape[0])
    print("Number of test samples for unscreened model:", test_df_unscreened.shape[0])
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    test_results = defaultdict(dict)
    test_results['iteration'] = args.max_iter
    for i in range(args.max_iter):
        print("#################################")
        print("Testing model", i)
        print("#################################\n")
        print("Testing screened model...")
        test_loss, test_auc, test_ap, test_f1, test_report, precision_99, precision_95, precision_90, f_max, t_max, true_y_list, pred_y_list = test_external(models_screened[i], test_df_screened, criterion, args.model, TEXTs_screened[i], nlp, device)
        print(f'Test Loss: {test_loss:.3f} | Test precision at 0.99: {precision_99*100:.2f}%')
        print(f'Test precision at 0.95: {precision_95*100:.2f}% | Test precision at 0.90: {precision_90*100:.2f}%')
        print(f'Test AUC: {test_auc:.2f} | Test AP: {test_ap*100:.2f}%')
        print(f'Test F-max: {f_max*100:.2f}% | Test threshold: {t_max:.2f}')
        print("\n")
        
        # save_results
        test_results['screened']['pred_y_' + str(i)] = pred_y_list
        test_results['screened']['precision_99_' + str(i)] = precision_99
        test_results['screened']['precision_95_' + str(i)] = precision_95
        test_results['screened']['precision_90_' + str(i)] = precision_90
        test_results['screened']['f-max_' + str(i)] = f_max
        test_results['screened']['t-max_' + str(i)] = t_max
        test_results['screened']['AUC_' + str(i)] = test_auc
        test_results['screened']['AUPR_' + str(i)] = test_ap
        
        print("Testing unscreened model...")
        test_loss, test_auc, test_ap, test_f1, test_report, precision_99, precision_95, precision_90, f_max, t_max, true_y_list, pred_y_list = test_external(models_unscreened[i], test_df_unscreened, criterion, args.model, TEXTs_unscreened[i], nlp, device)
        print(f'Test Loss: {test_loss:.3f} | Test precision at 0.99: {precision_99*100:.2f}%')
        print(f'Test precision at 0.95: {precision_95*100:.2f}% | Test precision at 0.90: {precision_90*100:.2f}%')
        print(f'Test AUC: {test_auc:.2f} | Test AP: {test_ap*100:.2f}%')
        print(f'Test F-max: {f_max*100:.2f}% | Test threshold: {t_max:.2f}')
        print("\n")

        # save_results
        test_results['unscreened']['pred_y_' + str(i)] = pred_y_list
        test_results['unscreened']['precision_99_' + str(i)] = precision_99
        test_results['unscreened']['precision_95_' + str(i)] = precision_95
        test_results['unscreened']['precision_90_' + str(i)] = precision_90
        test_results['unscreened']['f-max_' + str(i)] = f_max
        test_results['unscreened']['t-max_' + str(i)] = t_max
        test_results['unscreened']['AUC_' + str(i)] = test_auc
        test_results['unscreened']['AUPR_' + str(i)] = test_ap
    

    rand_str = np.random.randint(100000)
    prefix = args.model + "_" + args.select_method + "_" + str(args.top_num) + "_" + str(args.init_pos) + "_" + str(args.init_unscr_neg) + "_" + str(args.init_val) + "_" + str(args.sharing_pos) + "_" + str(args.n_epochs)
    print("save results...")
    with open("../results/results_" + prefix  + ".json", "w") as f:
        json.dump(test_results, f)
    test_df_screened.to_csv("../results/detail/test_df_screened_" + prefix + ".csv", index=False)
    test_df_unscreened.to_csv("../results/detail/test_df_unscreened_" + prefix + ".csv", index=False)
    copyfile("../data/iteration_" + str(args.max_iter-1) + "/train_screened_df.csv", "../results/detail/train_screened_df_" + prefix + ".csv")
    copyfile("../data/iteration_" + str(args.max_iter-1) + "/train_unscreened_df.csv", "../results/detail/train_unscreened_df_" + prefix + ".csv")
    