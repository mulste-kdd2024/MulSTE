import os
import sys
import copy
import csv

from lib import utils
from lib.figure_plot import train_loss_plot, valid_loss_plot, pred_plot
from lib.utils import log_string
# from lib.utils import generate_dataset, generate_dataset_for_each_datatype, Dataset
from lib.utils_fix_normalization_zz import generate_dataset, generate_dataset_for_each_datatype, Dataset
from lib.utils import masked_MAE, masked_RMSE, masked_MAPE, smape, masked_sMAPE
from lib.utils import l1_loss, l2_loss
from lib.utils import save_metrics
from lib.graph_utils import calculate_laplacian_with_self_loop, multi_graph_construction
from model.MulSTE_model_zz import MulSTE
from model.fine_tuned_bert.Model_Fine_Tuning import Model_Fine_Tuning

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import time
import sys
import datetime
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm 


import argparse
import configparser
import json

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, required=False, default='./config/data_zz.conf')
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)

# [file]
parser.add_argument('--d_path', type=str, default=config['file']['d_path'])
parser.add_argument('--s_path', type=str, default=config['file']['s_path'])
parser.add_argument('--ds_path', type=str, default=config['file']['ds_path'])
parser.add_argument('--outbreak_path', type=str, default=config['file']['outbreak_path'])
parser.add_argument('--risk_path', type=str, default=config['file']['risk_path'])
parser.add_argument('--news_fine_tuning_path', type=str, default=config['file']['news_fine_tuning_path'])
parser.add_argument('--news_path', type=str, default=config['file']['news_path'])
parser.add_argument('--daily_news_text_labeled_full_date_path', type=str, default=config['file']['daily_news_text_labeled_full_date_path'])
parser.add_argument('--festival_path', type=str, default=config['file']['festival_path'])

parser.add_argument('--adj_distance_path', type=str, default=config['file']['adj_distance_path'])
parser.add_argument('--adj_neighbor_path', type=str, default=config['file']['adj_neighbor_path'])
parser.add_argument('--adj_road_sim_path', type=str, default=config['file']['adj_road_sim_path'])
parser.add_argument('--adj_crowd_sim_path', type=str, default=config['file']['adj_crowd_sim_path'])

parser.add_argument('--pre_trained_bert_path', type=str, default=config['file']['pre_trained_bert_path'])
parser.add_argument('--fine_tuned_bert_path', type=str, default=config['file']['fine_tuned_bert_path'])


# [data]
parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
parser.add_argument('--total_seq_len', type=int, default=config['data']['total_seq_len'])
parser.add_argument('--T', type=int, default=config['data']['T'])
parser.add_argument('--l', type=int, default=config['data']['l'])
parser.add_argument('--train_ratio', type=float, default=config['data']['train_ratio'])
parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])

# [train]
parser.add_argument('--cuda', type=int, default=config['train']['cuda'])
parser.add_argument('--seed', type=int, default=config['train']['seed'])
parser.add_argument('--max_epoch', type=int, default=config['train']['max_epoch'])
parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
parser.add_argument('--early_stop', type=eval, default=config['train']['early_stop'])
parser.add_argument('--early_stop_patience', type=int, default=config['train']['early_stop_patience'])
parser.add_argument('--learning_rate', type=float, default=config['train']['learning_rate'])
parser.add_argument('--lr_decay_steps', type=json.loads, default=config['train']['lr_decay_steps'])
parser.add_argument('--lr_decay_rate', type=float, default=config['train']['lr_decay_rate'])
parser.add_argument('--loss_d_weight', type=float, default=config['train']['loss_d_weight'])
parser.add_argument('--loss_s_weight', type=float, default=config['train']['loss_s_weight'])

# [param]
parser.add_argument('--M', type=int, default=config['param']['M'])
parser.add_argument('--selected_M_d', type=int, default=config['param']['selected_M_d'])
parser.add_argument('--m_args_list_d', type=json.loads, action='append', default=config['param']['m_args_list_d'])
parser.add_argument('--selected_M_s', type=int, default=config['param']['selected_M_s'])
parser.add_argument('--m_args_list_s', type=json.loads, action='append', default=config['param']['m_args_list_s'])
parser.add_argument('--selected_M_ab', type=int, default=config['param']['selected_M_ab'])
parser.add_argument('--m_args_list_ab', type=json.loads, action='append', default=config['param']['m_args_list_ab'])
parser.add_argument('--input_dim', type=int, default=config['param']['input_dim'])
parser.add_argument('--hidden_dim', type=int, default=config['param']['hidden_dim'])
parser.add_argument('--feature_seq_len', type=int, default=config['param']['feature_seq_len'])
parser.add_argument('--target_seq_len', type=int, default=config['param']['target_seq_len'])

args = parser.parse_args()

current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'log', current_time)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
args.log_dir = log_dir

log = open(args.log_dir+'/train_valid_test_log_{}.txt'.format(current_time), 'a+') 


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(timedelta(seconds=elapsed_rounded))


def train(num_epoch, model, adj, train_dataloader, valid_dataloader, train_and_valid_var, std_d, mean_d, std_s, mean_s, with_or_without_interaction, with_or_without_event):

    (train_inputs_d, train_inputs_s, train_target_d, train_target_s, train_daily_news_input_ids, train_daily_news_token_type_ids, train_daily_news_attention_mask, train_daily_valid_news_mask, train_inputs_outbreak, train_inputs_risk, train_inputs_abnormal_news_num, train_festival_feature, train_festival_target,
    valid_inputs_d, valid_inputs_s, valid_target_d, valid_target_s, valid_daily_news_input_ids, valid_daily_news_token_type_ids, valid_daily_news_attention_mask, valid_daily_valid_news_mask, valid_inputs_outbreak, valid_inputs_risk, valid_inputs_abnormal_news_num, valid_festival_feature, valid_festival_target,) = train_and_valid_var    
    
    # print("Running Train...")
    log_string(log, "Running Train...")
    
    device = 'cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu' # 'cuda'/'gpu'  or 'cpu'
    # print('device=', device)
    log_string(log, "device={}".format(device))
    
    model = model.to(device)
    
    num_epoch = num_epoch

    lr_init = args.learning_rate
    lr_decay_steps = args.lr_decay_steps
    lr_decay_rate = args.lr_decay_rate
    
    best_loss = float('inf')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, eps = 1e-8) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = lr_decay_steps, gamma = lr_decay_rate)
    
    epoch_train_loss_d_list = []
    epoch_train_loss_s_list = []
    epoch_train_loss_list = []
    epoch_valid_loss_d_list = [] 
    epoch_valid_loss_s_list = []
    epoch_valid_loss_list = []
    for epoch in range(num_epoch):
        
        log_string(log, "——————Epoch {:} / {:} ——————".format(epoch + 1,num_epoch))
        
        start = time.time()
        model.train() 
        train_loss_d_list = [] 
        train_loss_s_list = []
        train_loss_list = []
        # batch loop:
        loop = tqdm(enumerate(train_dataloader), total =len(train_dataloader), desc='Train progress bar', file=sys.stdout) 
        for i, train_indices in loop: 
            
            train_inputs_d_batch = train_inputs_d[train_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
            train_inputs_s_batch = train_inputs_s[train_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
            # print(train_inputs_d.shape)
            
            train_target_d_batch = train_target_d[train_indices].to(device) # (batch_size, target_seq_len, num_nodes, output_dim)
            train_target_s_batch = train_target_s[train_indices].to(device) # (batch_size, target_seq_len, num_nodes, output_dim)
            # print(train_target_d.shape)
            
            train_daily_news_input_ids_batch = train_daily_news_input_ids[train_indices].to(device)           # (batch_size, feature_seq_len, num_news, num_tokens)
            train_daily_news_token_type_ids_batch = train_daily_news_token_type_ids[train_indices].to(device) # (batch_size, feature_seq_len, num_news, num_tokens)
            train_daily_news_attention_mask_batch = train_daily_news_attention_mask[train_indices].to(device) # (batch_size, feature_seq_len, num_news, num_tokens)
            train_daily_valid_news_mask_batch = train_daily_valid_news_mask[train_indices].to(device)
             
            train_inputs_outbreak_batch = train_inputs_outbreak[train_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
            train_inputs_risk_batch = train_inputs_risk[train_indices].to(device)         # (batch_size, feature_seq_len, num_nodes, input_dim)
            train_inputs_abnormal_news_num_batch = train_inputs_abnormal_news_num[train_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
            
            train_festival_feature_batch = train_festival_feature[train_indices].to(device)
            train_festival_target_batch = train_festival_target[train_indices].to(device)
            
            optimizer.zero_grad() 
            model.zero_grad() 

            last_output_d_batch, last_output_s_batch,_,_,_,_,_,_,_,_,_,_,_,_ = model(
                adj = adj,
                inputs_d = train_inputs_d_batch,
                inputs_s = train_inputs_s_batch,
                daily_news_input_ids = train_daily_news_input_ids_batch,
                daily_news_token_type_ids = train_daily_news_token_type_ids_batch,
                daily_news_attention_mask = train_daily_news_attention_mask_batch,
                daily_valid_news_mask = train_daily_valid_news_mask_batch,
                inputs_outbreak = train_inputs_outbreak_batch,
                inputs_risk = train_inputs_risk_batch,
                inputs_abnormal_news_num = train_inputs_abnormal_news_num_batch,
                inputs_festival_feature = train_festival_feature_batch,
                inputs_festival_target = train_festival_target_batch,
                with_or_without_interaction = with_or_without_interaction,
                with_or_without_event = with_or_without_event, 
            )
            
            last_output_d_batch = last_output_d_batch * std_d + mean_d
            last_output_s_batch = last_output_s_batch * std_s + mean_s
            
            loss_d = l1_loss(last_output_d_batch, train_target_d_batch)
            loss_s = l1_loss(last_output_s_batch, train_target_s_batch)
            loss = args.loss_d_weight * loss_d + args.loss_s_weight * loss_s            
            
            train_loss_d_list.append(loss_d.item())
            train_loss_s_list.append(loss_s.item())
            train_loss_list.append(loss.item())
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

            optimizer.step() 

        scheduler.step() 
 
        epoch_train_loss_d_list.append(np.mean(train_loss_d_list))
        epoch_train_loss_s_list.append(np.mean(train_loss_s_list))
        epoch_train_loss_list.append(np.mean(train_loss_list))
        
        if np.mean(train_loss_list) > 1e6:
            log_string(log, "Gradient explosion detected. Ending...")
            break
        
        training_time = format_time(time.time() - start)

        log_string(log, "Epoch: {}, Train Demand Loss: {}, Train Supply Loss: {}, Train Loss: {}, Lr: {}, Time: {}".format(epoch, np.mean(train_loss_d_list), np.mean(train_loss_s_list), np.mean(train_loss_list), optimizer.state_dict()['param_groups'][0]['lr'], training_time))
        

        log_string(log, "Running Validation...")
        
        model.eval()
        valid_loss_d_list = [] 
        valid_loss_s_list = []
        valid_loss_list = []
        # batch loop:
        loop = tqdm(enumerate(valid_dataloader), total =len(valid_dataloader), desc='Valid progress bar', file=sys.stdout)
        for i, valid_indices in loop:

            ## 
            valid_inputs_d_batch = valid_inputs_d[valid_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
            valid_inputs_s_batch = valid_inputs_s[valid_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
            
            ## 
            valid_target_d_batch = valid_target_d[valid_indices].to(device) # (batch_size, target_seq_len, num_nodes, output_dim)
            valid_target_s_batch = valid_target_s[valid_indices].to(device) # (batch_size, target_seq_len, num_nodes, output_dim)
            
            ## 
            valid_daily_news_input_ids_batch = valid_daily_news_input_ids[valid_indices].to(device)           # (batch_size, feature_seq_len, num_news, num_tokens)
            valid_daily_news_token_type_ids_batch = valid_daily_news_token_type_ids[valid_indices].to(device) # (batch_size, feature_seq_len, num_news, num_tokens)
            valid_daily_news_attention_mask_batch = valid_daily_news_attention_mask[valid_indices].to(device) # (batch_size, feature_seq_len, num_news, num_tokens)
            valid_daily_valid_news_mask_batch = valid_daily_valid_news_mask[valid_indices].to(device)            
            
            ## 
            valid_inputs_outbreak_batch = valid_inputs_outbreak[valid_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
            valid_inputs_risk_batch = valid_inputs_risk[valid_indices].to(device)         # (batch_size, feature_seq_len, num_nodes, input_dim)
            valid_inputs_abnormal_news_num_batch = valid_inputs_abnormal_news_num[valid_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
            
            ## 
            valid_festival_feature_batch = valid_festival_feature[valid_indices].to(device)
            valid_festival_target_batch = valid_festival_target[valid_indices].to(device)
            
            with torch.no_grad():
                last_output_d_batch, last_output_s_batch,_,_,_,_,_,_,_,_,_,_,_,_ = model(
                    adj = adj,
                    inputs_d = valid_inputs_d_batch,
                    inputs_s = valid_inputs_s_batch,
                    daily_news_input_ids = valid_daily_news_input_ids_batch,
                    daily_news_token_type_ids = valid_daily_news_token_type_ids_batch,
                    daily_news_attention_mask = valid_daily_news_attention_mask_batch,
                    daily_valid_news_mask = valid_daily_valid_news_mask_batch,
                    inputs_outbreak = valid_inputs_outbreak_batch,
                    inputs_risk = valid_inputs_risk_batch,
                    inputs_abnormal_news_num = valid_inputs_abnormal_news_num_batch,
                    inputs_festival_feature = valid_festival_feature_batch,
                    inputs_festival_target = valid_festival_target_batch,
                    with_or_without_interaction = with_or_without_interaction,
                    with_or_without_event = with_or_without_event,      
                )
                
                last_output_d_batch = last_output_d_batch * std_d + mean_d
                last_output_s_batch = last_output_s_batch * std_s + mean_s
                
                loss_d = l1_loss(last_output_d_batch, valid_target_d_batch)
                loss_s = l1_loss(last_output_s_batch, valid_target_s_batch)

                loss = args.loss_d_weight * loss_d + args.loss_s_weight * loss_s            
                
            valid_loss_d_list.append(loss_d.item())
            valid_loss_s_list.append(loss_s.item())
            valid_loss_list.append(loss.item())
        
        epoch_valid_loss_d_list.append(np.mean(valid_loss_d_list))
        epoch_valid_loss_s_list.append(np.mean(valid_loss_s_list))
        epoch_valid_loss_list.append(np.mean(valid_loss_list))    
        
        # print("Epoch: {}, Valid Demand Loss: {}, Valid Supply Loss: {}, Valid Loss: {}".format(epoch, np.mean(valid_loss_d_list), np.mean(valid_loss_s_list), np.mean(valid_loss_list))) 
        log_string(log, "Epoch: {}, Valid Demand Loss: {}, Valid Supply Loss: {}, Valid Loss: {}".format(epoch, np.mean(valid_loss_d_list), np.mean(valid_loss_s_list), np.mean(valid_loss_list)))
        
        # early stop and save the best state 
        # early stop
        if np.mean(valid_loss_list) < best_loss:
            best_loss = np.mean(valid_loss_list)
            not_improved_count = 0
            best_state = True
        else:
            not_improved_count += 1
            best_state = False
        if args.early_stop:
            if not_improved_count == args.early_stop_patience: 
                log_string(log,"Validation performance didn\'t improve for {} epochs. "
                                "Training stops.".format(args.early_stop_patience))
                break 
        # save the best state
        if best_state == True: 
            log_string(log,'#####################Current best model saved!#####################')
            best_model = copy.deepcopy(model.state_dict())  
            
    #save the best model to file
    torch.save(best_model, args.log_dir+'/mulste_{}.pth'.format(current_time))
    log_string(log, "Saving current best model to " + args.log_dir+'/mulste_{}.pth'.format(current_time))  
        
    epoch_train_loss_d = torch.Tensor(epoch_train_loss_d_list)
    epoch_train_loss_s = torch.Tensor(epoch_train_loss_s_list)
    epoch_train_loss = torch.Tensor(epoch_train_loss_list)
    epoch_valid_loss_d = torch.Tensor(epoch_valid_loss_d_list)
    epoch_valid_loss_s = torch.Tensor(epoch_valid_loss_s_list)
    epoch_valid_loss = torch.Tensor(epoch_valid_loss_list)
    
    # torch.save(epoch_train_loss_d, args.log_dir + '/epoch_train_loss_d_{}.pth'.format(current_time)) # list -> tensor
    # torch.save(epoch_train_loss_s, args.log_dir + '/epoch_train_loss_s_{}.pth'.format(current_time))
    # torch.save(epoch_train_loss, args.log_dir + '/epoch_train_loss_{}.pth'.format(current_time))
    # torch.save(epoch_valid_loss_d, args.log_dir + '/epoch_valid_loss_d_{}.pth'.format(current_time))
    # torch.save(epoch_valid_loss_s, args.log_dir + '/epoch_valid_loss_s_{}.pth'.format(current_time))
    # torch.save(epoch_valid_loss, args.log_dir + '/epoch_valid_loss_{}.pth'.format(current_time))
    train_loss_plot(args, current_time, epoch_train_loss_d, epoch_train_loss_s, epoch_train_loss)
    valid_loss_plot(args, current_time, epoch_valid_loss_d, epoch_valid_loss_s, epoch_valid_loss)
    
    return best_model, epoch_train_loss_d, epoch_train_loss_s, epoch_train_loss, epoch_valid_loss_d, epoch_valid_loss_s, epoch_valid_loss


def test(model, model_name, adj, test_dataloader, test_var, std_d, mean_d, std_s, mean_s, drop_last, with_or_without_interaction, with_or_without_event,):
    
    """
    num_samples
    num_nodes
    
    return 
    outputs_d (num_samples, num_nodes)
    outputs_s (num_samples, num_nodes)
    targets_d (num_samples, num_nodes)
    targets_s (num_samples, num_nodes)
    """
    (test_inputs_d, test_inputs_s, test_target_d, test_target_s, test_daily_news_input_ids, test_daily_news_token_type_ids, test_daily_news_attention_mask, test_daily_valid_news_mask, test_inputs_outbreak, test_inputs_risk, test_inputs_abnormal_news_num, test_festival_feature, test_festival_target,) = test_var
    
    
    num_nodes = 71
    feature_seq_len = test_inputs_d.shape[1]
    target_seq_len = test_target_d.shape[1]
    
    if drop_last == True:
        num_samples = test_dataloader.batch_size * len(test_dataloader) # batch_size * batch_num [drop the last batch in test_dataloader]

    elif drop_last == False:
        num_samples = len(Dataset(indices,'test', args)) # initial samples number in test dataset [not drop the last batch in test_dataloader]

    # print("Running Test...")
    log_string(log, "Running Test...")
    
    device = 'cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu' # 'cuda'/'gpu'  or 'cpu'
    # print('device=', device)
    log_string(log, "device={}".format(device))
    
    # global model 
    model = model.to(device)
    
    model.eval()
    
    # metrics [MAE/RMSE/MAPE]
    MAE_d_list = []
    MAE_s_list = []
    RMSE_d_list = []
    RMSE_s_list = []
    MAPE_d_list = []
    MAPE_s_list = []
    sMAPE_d_list = []
    sMAPE_s_list = []
    m_sMAPE_d_list = []
    m_sMAPE_s_list = []    
    
    attention_weight1_d_list = []
    full_adaptive_adj1_d_list = [] 
    attention_weight2_d_list = [] 
    full_adaptive_adj2_d_list = []
     
    attention_weight1_s_list = [] 
    full_adaptive_adj1_s_list = [] 
    attention_weight2_s_list = [] 
    full_adaptive_adj2_s_list = [] 
    
    attention_weight1_ab_list = [] 
    full_adaptive_adj1_ab_list = [] 
    attention_weight2_ab_list = [] 
    full_adaptive_adj2_ab_list = [] 
    
    outputs_d_list = []
    outputs_s_list = []
    targets_d_list = []
    targets_s_list = []
    
    # Batch loop:
    loop = tqdm(enumerate(test_dataloader), total =len(test_dataloader), desc='Test progress bar', file=sys.stdout)
    for i, test_indices in loop:

        ## test_feature = test_feature.to(device)
        test_inputs_d_batch = test_inputs_d[test_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
        test_inputs_s_batch = test_inputs_s[test_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
        
        ## test_target = test_target.to(device)
        test_target_d_batch = test_target_d[test_indices].to(device) # (batch_size, target_seq_len, num_nodes, output_dim)
        test_target_s_batch = test_target_s[test_indices].to(device) # (batch_size, target_seq_len, num_nodes, output_dim)
        
        ## 
        test_daily_news_input_ids_batch = test_daily_news_input_ids[test_indices].to(device)           # (batch_size, feature_seq_len, num_news, num_tokens)
        test_daily_news_token_type_ids_batch = test_daily_news_token_type_ids[test_indices].to(device) # (batch_size, feature_seq_len, num_news, num_tokens)
        test_daily_news_attention_mask_batch = test_daily_news_attention_mask[test_indices].to(device) # (batch_size, feature_seq_len, num_news, num_tokens)
        test_daily_valid_news_mask_batch = test_daily_valid_news_mask[test_indices].to(device)
        
        ## 
        test_inputs_outbreak_batch = test_inputs_outbreak[test_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
        test_inputs_risk_batch = test_inputs_risk[test_indices].to(device)         # (batch_size, feature_seq_len, num_nodes, input_dim)
        test_inputs_abnormal_news_num_batch = test_inputs_abnormal_news_num[test_indices].to(device) # (batch_size, feature_seq_len, num_nodes, input_dim)
        
        ## 
        test_festival_feature_batch = test_festival_feature[test_indices].to(device)
        test_festival_target_batch = test_festival_target[test_indices].to(device)
        
        with torch.no_grad(): 
            last_output_d_batch, last_output_s_batch, attention_weight1_d_batch, full_adaptive_adj1_d_batch, attention_weight2_d_batch, full_adaptive_adj2_d_batch, attention_weight1_s_batch, full_adaptive_adj1_s_batch, attention_weight2_s_batch, full_adaptive_adj2_s_batch, attention_weight1_ab_batch, full_adaptive_adj1_ab_batch, attention_weight2_ab_batch, full_adaptive_adj2_ab_batch = model(
                adj = adj,
                inputs_d = test_inputs_d_batch,
                inputs_s = test_inputs_s_batch,
                daily_news_input_ids = test_daily_news_input_ids_batch,
                daily_news_token_type_ids = test_daily_news_token_type_ids_batch,
                daily_news_attention_mask = test_daily_news_attention_mask_batch,
                daily_valid_news_mask = test_daily_valid_news_mask_batch,
                inputs_outbreak = test_inputs_outbreak_batch,
                inputs_risk = test_inputs_risk_batch,
                inputs_abnormal_news_num = test_inputs_abnormal_news_num_batch,
                inputs_festival_feature = test_festival_feature_batch,
                inputs_festival_target = test_festival_target_batch,
                with_or_without_interaction = with_or_without_interaction,
                with_or_without_event = with_or_without_event, 
            )
        
        last_output_d_batch = last_output_d_batch * std_d + mean_d
        last_output_s_batch = last_output_s_batch * std_s + mean_s
        
        
        MAE_d_list.append(masked_MAE(last_output_d_batch, test_target_d_batch).cpu().numpy()) 
        MAE_s_list.append(masked_MAE(last_output_s_batch, test_target_s_batch).cpu().numpy())
        
        RMSE_d_list.append(masked_RMSE(last_output_d_batch, test_target_d_batch).cpu().numpy())
        RMSE_s_list.append(masked_RMSE(last_output_s_batch, test_target_s_batch).cpu().numpy())

        MAPE_d_list.append(masked_MAPE(last_output_d_batch, test_target_d_batch, mask_value = 0).cpu().numpy())
        MAPE_s_list.append(masked_MAPE(last_output_s_batch, test_target_s_batch, mask_value = 0).cpu().numpy())
        
        sMAPE_d_list.append(masked_sMAPE(last_output_d_batch, test_target_d_batch, mask_value=0).cpu().numpy())
        sMAPE_s_list.append(masked_sMAPE(last_output_s_batch, test_target_s_batch, mask_value=0).cpu().numpy())

        sMAPE_d_list.append(smape(last_output_d_batch, test_target_d_batch).cpu().numpy())
        sMAPE_s_list.append(smape(last_output_s_batch, test_target_s_batch).cpu().numpy())
        
        m_sMAPE_d_list.append(masked_sMAPE(last_output_d_batch, test_target_d_batch, mask_value=0).cpu().numpy())
        m_sMAPE_s_list.append(masked_sMAPE(last_output_s_batch, test_target_s_batch, mask_value=0).cpu().numpy())
        
        attention_weight1_d_list.append(attention_weight1_d_batch)
        full_adaptive_adj1_d_list.append(full_adaptive_adj1_d_batch)
        attention_weight2_d_list.append(attention_weight2_d_batch)
        full_adaptive_adj2_d_list.append(full_adaptive_adj2_d_batch)
        
        attention_weight1_s_list.append(attention_weight1_s_batch)
        full_adaptive_adj1_s_list.append(full_adaptive_adj1_s_batch)
        attention_weight2_s_list.append(attention_weight2_s_batch)
        full_adaptive_adj2_s_list.append(full_adaptive_adj2_s_batch) 
        
        attention_weight1_ab_list.append(attention_weight1_ab_batch)
        full_adaptive_adj1_ab_list.append(full_adaptive_adj1_ab_batch)
        attention_weight2_ab_list.append(attention_weight2_ab_batch)
        full_adaptive_adj2_ab_list.append(full_adaptive_adj2_ab_batch)
        
        attention_weight1_d = torch.stack(attention_weight1_d_list)
        full_adaptive_adj1_d = torch.stack(full_adaptive_adj1_d_list)
        attention_weight2_d = torch.stack(attention_weight2_d_list)
        full_adaptive_adj2_d = torch.stack(full_adaptive_adj2_d_list)
        torch.save(attention_weight1_d, args.log_dir + '/attention_weight1_d_'+ model_name +'_{}.pth'.format(current_time))
        torch.save(full_adaptive_adj1_d, args.log_dir + '/full_adaptive_adj1_d_'+ model_name +'_{}.pth'.format(current_time))
        torch.save(attention_weight2_d, args.log_dir + '/attention_weight2_d_'+ model_name +'_{}.pth'.format(current_time))
        torch.save(full_adaptive_adj2_d, args.log_dir + '/full_adaptive_adj2_d_'+ model_name +'_{}.pth'.format(current_time))
        
        attention_weight1_s = torch.stack(attention_weight1_s_list)
        full_adaptive_adj1_s = torch.stack(full_adaptive_adj1_s_list)
        attention_weight2_s = torch.stack(attention_weight2_s_list)
        full_adaptive_adj2_s = torch.stack(full_adaptive_adj2_s_list)
        torch.save(attention_weight1_s, args.log_dir + '/attention_weight1_s_'+ model_name +'_{}.pth'.format(current_time))
        torch.save(full_adaptive_adj1_s, args.log_dir + '/full_adaptive_adj1_s_'+ model_name +'_{}.pth'.format(current_time))
        torch.save(attention_weight2_s, args.log_dir + '/attention_weight2_s_'+ model_name +'_{}.pth'.format(current_time))
        torch.save(full_adaptive_adj2_s, args.log_dir + '/full_adaptive_adj2_s_'+ model_name +'_{}.pth'.format(current_time))
        
        attention_weight1_ab = torch.stack(attention_weight1_ab_list)
        full_adaptive_adj1_ab = torch.stack(full_adaptive_adj1_ab_list)
        attention_weight2_ab = torch.stack(attention_weight2_ab_list)
        full_adaptive_adj2_ab = torch.stack(full_adaptive_adj2_ab_list)
        torch.save(attention_weight1_ab, args.log_dir + '/attention_weight1_ab_'+ model_name +'_{}.pth'.format(current_time))
        torch.save(full_adaptive_adj1_ab, args.log_dir + '/full_adaptive_adj1_ab_'+ model_name +'_{}.pth'.format(current_time))
        torch.save(attention_weight2_ab, args.log_dir + '/attention_weight2_ab_'+ model_name +'_{}.pth'.format(current_time))
        torch.save(full_adaptive_adj2_ab, args.log_dir + '/full_adaptive_adj2_ab_'+ model_name +'_{}.pth'.format(current_time))

        # final appended list shape: (batch_num, batch_size, num_nodes)
        if drop_last == True: # drop the last batch in test_dataloader
            outputs_d_list.append(last_output_d_batch.cpu().numpy().tolist()) # tensor -> ndarray -> list
            outputs_s_list.append(last_output_s_batch.cpu().numpy().tolist()) # tensor -> ndarray -> list
            targets_d_list.append(test_target_d_batch.cpu().numpy().tolist()) # tensor -> ndarray -> list
            targets_s_list.append(test_target_s_batch.cpu().numpy().tolist()) # tensor -> ndarray -> list
        
        # final appended list shape: (num_samples, num_nodes) 
        elif drop_last == False: # not drop the last batch in test_dataloader
            for sample_i in last_output_d_batch.squeeze().cpu().numpy().reshape(-1,num_nodes).tolist(): # (batch_size, num_nodes, 1) -> (batch_size, num_nodes) -> (-1, num_nodes); tensor -> ndarray -> list
                outputs_d_list.append(sample_i)
            for sample_i in last_output_s_batch.squeeze().cpu().numpy().reshape(-1,num_nodes).tolist(): # (batch_size, num_nodes, 1) -> (batch_size, num_nodes) -> (-1, num_nodes); tensor -> ndarray -> list
                outputs_s_list.append(sample_i)
            for sample_i in test_target_d_batch.squeeze().cpu().numpy().reshape(-1,num_nodes).tolist(): # (batch_size, num_nodes, 1) -> (batch_size, num_nodes) -> (-1, num_nodes); tensor -> ndarray -> list
                targets_d_list.append(sample_i)
            for sample_i in test_target_s_batch.squeeze().cpu().numpy().reshape(-1,num_nodes).tolist():  # (batch_size, num_nodes, 1) -> (batch_size, num_nodes) -> (-1, num_nodes); tensor -> ndarray -> list
                targets_s_list.append(sample_i)
                
    # print("Test Demand MAE: {}, Test Supply MAE: {}".format(np.mean(MAE_d_list),np.mean(MAE_s_list)))
    log_string(log, "Test Demand MAE: {}, Test Supply MAE: {}".format(np.mean(MAE_d_list),np.mean(MAE_s_list)))
    # print("Test Demand RMSE: {}, Test Supply RMSE: {}".format(np.mean(RMSE_d_list),np.mean(RMSE_s_list)))
    log_string(log, "Test Demand RMSE: {}, Test Supply RMSE: {}".format(np.mean(RMSE_d_list),np.mean(RMSE_s_list)))
    # print("Test Demand MAPE: {}, Test Supply MAPE: {}".format(np.mean(MAPE_d_list),np.mean(MAPE_s_list)))
    log_string(log, "Test Demand MAPE: {}, Test Supply MAPE: {}".format(np.mean(MAPE_d_list),np.mean(MAPE_s_list)))
    # print("Test Demand sMAPE: {}, Test Supply sMAPE: {}".format(np.mean(sMAPE_d_list),np.mean(sMAPE_s_list)))
    log_string(log, "Test Demand sMAPE: {}, Test Supply sMAPE: {}".format(np.mean(sMAPE_d_list),np.mean(sMAPE_s_list)))
    # print("Test Demand m_sMAPE: {}, Test Supply m_sMAPE: {}".format(np.mean(m_sMAPE_d_list),np.mean(m_sMAPE_s_list)))
    log_string(log, "Test Demand m_sMAPE: {}, Test Supply m_sMAPE: {}".format(np.mean(m_sMAPE_d_list),np.mean(m_sMAPE_s_list)))
    
    save_metrics(args, model_name = model_name, d_or_s = 'd', MAE = np.mean(MAE_d_list), RMSE = np.mean(RMSE_d_list), sMAPE = np.mean(m_sMAPE_d_list))
    save_metrics(args, model_name = model_name, d_or_s = 's', MAE = np.mean(MAE_s_list), RMSE = np.mean(RMSE_s_list), sMAPE = np.mean(m_sMAPE_s_list))
    
    outputs_d = torch.Tensor(outputs_d_list).reshape(num_samples, target_seq_len, num_nodes)
    outputs_s = torch.Tensor(outputs_s_list).reshape(num_samples, target_seq_len, num_nodes)
    targets_d = torch.Tensor(targets_d_list).reshape(num_samples, target_seq_len, num_nodes)
    targets_s = torch.Tensor(targets_s_list).reshape(num_samples, target_seq_len, num_nodes)
    
    torch.save(outputs_d, args.log_dir + '/outputs_d_'+ model_name +'_{}.pth'.format(current_time))
    torch.save(outputs_s, args.log_dir + '/outputs_s_'+ model_name +'_{}.pth'.format(current_time))
    torch.save(targets_d, args.log_dir + '/targets_d_'+ model_name +'_{}.pth'.format(current_time))
    torch.save(targets_s, args.log_dir + '/targets_s_'+ model_name +'_{}.pth'.format(current_time))
    
    # print(num_samples)
    pred_plot(args, current_time, range(0, num_samples), outputs_d, targets_d, outputs_s, targets_s)
    
    return outputs_d, outputs_s, targets_d, targets_s


# def train():
if __name__ == '__main__':
    
    log_string(log, 'MulSTE--------------------------------------------------------------------------------------')
    #-------------------------------------------------------------------------------------
    adj = multi_graph_construction(args)
    ds_feature, ds_target, outbreak_feature, risk_feature, news_matrix_fine_tuning, daily_news_input_ids_feature, daily_news_token_type_ids_feature, daily_news_attention_mask_feature, daily_valid_news_mask_feature, abnormal_news_num_feature, festival_feature, festival_target = generate_dataset_for_each_datatype(args)
    
    
    mulste = MulSTE(M = args.M, selected_M_d = args.selected_M_d, m_args_list_d = args.m_args_list_d, 
                  selected_M_s = args.selected_M_s, m_args_list_s = args.m_args_list_s, 
                  selected_M_ab = args.selected_M_ab, m_args_list_ab = args.m_args_list_ab, 
                  fine_tuned_bert_path = args.fine_tuned_bert_path, 
                  input_dim = args.input_dim, hidden_dim = args.hidden_dim, 
                  feature_seq_len = args.feature_seq_len, target_seq_len = args.target_seq_len, 
                  with_or_without_interaction = 'with_interaction', with_or_without_event = 'with_event'  )
    
    #-------------------------------------------------------------------------------------
    indices = [i for i in range(args.total_seq_len - (args.T + args.T) + 1)] # indices are different from intervals
    train_indices = Dataset(indices,'train', args)
    valid_indices = Dataset(indices,'valid', args)
    test_indices = Dataset(indices,'test', args)
    
    # Training Dataset
    ## demand-supply sequence 
    train_inputs_d = ds_feature[train_indices][:,:,:,0:1]
    train_inputs_s = ds_feature[train_indices][:,:,:,1:2]
    train_target_d = ds_target[train_indices][:,:,:,0:1]
    train_target_s = ds_target[train_indices][:,:,:,1:2]
    ### normalization
    mean_d, std_d = torch.mean(train_inputs_d), torch.std(train_inputs_d)
    mean_s, std_s = torch.mean(train_inputs_s), torch.std(train_inputs_s)
    train_inputs_d = (train_inputs_d - mean_d) / std_d
    train_inputs_s = (train_inputs_s - mean_s) / std_s
    
    ## textual event
    train_daily_news_input_ids = daily_news_input_ids_feature[train_indices]                      # (batch_size, feature_seq_len, num_news, num_tokens)
    train_daily_news_token_type_ids = daily_news_token_type_ids_feature[train_indices]            # (batch_size, feature_seq_len, num_news, num_tokens)
    train_daily_news_attention_mask = daily_news_attention_mask_feature[train_indices]            # (batch_size, feature_seq_len, num_news, num_tokens)
    train_daily_valid_news_mask = daily_valid_news_mask_feature[train_indices] 
    
    ## numerical event 
    train_inputs_outbreak = outbreak_feature[train_indices]            # (batch_size, feature_seq_len, num_nodes, input_dim)
    train_inputs_risk = risk_feature[train_indices]                    # (batch_size, feature_seq_len, num_nodes, input_dim)
    train_inputs_abnormal_news_num = abnormal_news_num_feature[train_indices]              # (batch_size, feature_seq_len, num_nodes, input_dim) 
    ### normalization
    mean_outbreak, std_outbreak = torch.mean(train_inputs_outbreak), torch.std(train_inputs_outbreak)
    mean_risk, std_risk = torch.mean(train_inputs_risk), torch.std(train_inputs_risk)
    mean_abnormal_news_num, std_abnormal_news_num = torch.mean(train_inputs_abnormal_news_num), torch.std(train_inputs_abnormal_news_num)
    train_inputs_outbreak = (train_inputs_outbreak - mean_outbreak) / std_outbreak
    train_inputs_risk = (train_inputs_risk - mean_risk) / std_risk
    train_inputs_abnormal_news_num = (train_inputs_abnormal_news_num - mean_abnormal_news_num) / std_abnormal_news_num
    
    ## categorical event
    train_festival_feature = festival_feature[train_indices]
    train_festival_target = festival_target[train_indices]
    
    # Validation Dataset
    ## demand-supply sequence 
    valid_inputs_d = ds_feature[valid_indices][:,:,:,0:1]
    valid_inputs_s = ds_feature[valid_indices][:,:,:,1:2]
    valid_target_d = ds_target[valid_indices][:,:,:,0:1]
    valid_target_s = ds_target[valid_indices][:,:,:,1:2]
    ### normalization
    valid_inputs_d = (valid_inputs_d - mean_d) / std_d
    valid_inputs_s = (valid_inputs_s - mean_s) / std_s
    
    ## textual event
    valid_daily_news_input_ids = daily_news_input_ids_feature[valid_indices]          # (batch_size, feature_seq_len, num_news, num_tokens)
    valid_daily_news_token_type_ids = daily_news_token_type_ids_feature[valid_indices] # (batch_size, feature_seq_len, num_news, num_tokens)
    valid_daily_news_attention_mask = daily_news_attention_mask_feature[valid_indices] # (batch_size, feature_seq_len, num_news, num_tokens)
    valid_daily_valid_news_mask = daily_valid_news_mask_feature[valid_indices]            
            
    ## numerical event 
    valid_inputs_outbreak = outbreak_feature[valid_indices] # (batch_size, feature_seq_len, num_nodes, input_dim)
    valid_inputs_risk = risk_feature[valid_indices]         # (batch_size, feature_seq_len, num_nodes, input_dim)
    valid_inputs_abnormal_news_num = abnormal_news_num_feature[valid_indices] # (batch_size, feature_seq_len, num_nodes, input_dim)
    ### normalization
    valid_inputs_outbreak = (valid_inputs_outbreak - mean_outbreak) / std_outbreak
    valid_inputs_risk = (valid_inputs_risk - mean_risk) / std_risk
    valid_inputs_abnormal_news_num = (valid_inputs_abnormal_news_num - mean_abnormal_news_num) / std_abnormal_news_num
    
    ## categorical event
    valid_festival_feature = festival_feature[valid_indices]
    valid_festival_target = festival_target[valid_indices]
    
    # Testing Dataset
    ## demand-supply sequence 
    test_inputs_d = ds_feature[test_indices][:,:,:,0:1]
    test_inputs_s = ds_feature[test_indices][:,:,:,1:2]
    test_target_d = ds_target[test_indices][:,:,:,0:1]
    test_target_s = ds_target[test_indices][:,:,:,1:2]
    ### normalization
    test_inputs_d = (test_inputs_d - mean_d) / std_d
    test_inputs_s = (test_inputs_s - mean_s) / std_s
    
    ## textual event
    test_daily_news_input_ids = daily_news_input_ids_feature[test_indices]           # (batch_size, feature_seq_len, num_news, num_tokens)
    test_daily_news_token_type_ids = daily_news_token_type_ids_feature[test_indices] # (batch_size, feature_seq_len, num_news, num_tokens)
    test_daily_news_attention_mask = daily_news_attention_mask_feature[test_indices] # (batch_size, feature_seq_len, num_news, num_tokens)
    test_daily_valid_news_mask = daily_valid_news_mask_feature[test_indices]
    
    ## numerical event 
    test_inputs_outbreak = outbreak_feature[test_indices] # (batch_size, feature_seq_len, num_nodes, input_dim)
    test_inputs_risk = risk_feature[test_indices]         # (batch_size, feature_seq_len, num_nodes, input_dim)
    test_inputs_abnormal_news_num = abnormal_news_num_feature[test_indices] # (batch_size, feature_seq_len, num_nodes, input_dim)
    ### normalization
    test_inputs_outbreak = (test_inputs_outbreak - mean_outbreak) / std_outbreak
    test_inputs_risk = (test_inputs_risk - mean_risk) / std_risk
    test_inputs_abnormal_news_num = (test_inputs_abnormal_news_num - mean_abnormal_news_num) / std_abnormal_news_num
    
    ## categorical event
    test_festival_feature = festival_feature[test_indices]
    test_festival_target = festival_target[test_indices]

    train_and_valid_var = (train_inputs_d, train_inputs_s, train_target_d, train_target_s, train_daily_news_input_ids, train_daily_news_token_type_ids, train_daily_news_attention_mask, train_daily_valid_news_mask, train_inputs_outbreak, train_inputs_risk, train_inputs_abnormal_news_num, train_festival_feature, train_festival_target,
                        valid_inputs_d, valid_inputs_s, valid_target_d, valid_target_s, valid_daily_news_input_ids, valid_daily_news_token_type_ids, valid_daily_news_attention_mask, valid_daily_valid_news_mask, valid_inputs_outbreak, valid_inputs_risk, valid_inputs_abnormal_news_num, valid_festival_feature, valid_festival_target,)
    test_var  = (test_inputs_d, test_inputs_s, test_target_d, test_target_s, test_daily_news_input_ids, test_daily_news_token_type_ids, test_daily_news_attention_mask, test_daily_valid_news_mask, test_inputs_outbreak, test_inputs_risk, test_inputs_abnormal_news_num, test_festival_feature, test_festival_target,)
    
    ######################################################################################
    # train_dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset= range(len(train_indices)), batch_size=args.batch_size, shuffle=True, drop_last=True,)  
    # valid_dataloader
    valid_dataloader = torch.utils.data.DataLoader(dataset= range(len(valid_indices)),  batch_size=args.batch_size, shuffle=False, drop_last=True,)  
    # test_dataloader
    test_dataloader = torch.utils.data.DataLoader(dataset= range(len(test_indices)), batch_size=args.batch_size, shuffle=False, drop_last=True,)
    
    # train
    best_model, epoch_train_loss_d, epoch_train_loss_s, epoch_train_loss, epoch_valid_loss_d, epoch_valid_loss_s, epoch_valid_loss = train(args.max_epoch, mulste, adj, train_dataloader, valid_dataloader, train_and_valid_var, std_d, mean_d, std_s, mean_s, with_or_without_interaction = 'with_interaction', with_or_without_event = 'with_event') 
    mulste.load_state_dict(best_model) 

    # test
    outputs_d, outputs_s, targets_d, targets_s = test(mulste, 'MulSTE', adj, test_dataloader, test_var, std_d, mean_d, std_s, mean_s, drop_last = True, with_or_without_interaction = 'with_interaction', with_or_without_event = 'with_event')    