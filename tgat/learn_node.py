"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import sys
import random
import argparse

# from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from module import TGAN
# from graph import NeighborFinder
from graph import get_neighbor_finder
from data_processing import get_data_node_clf


def eval_epoch(src_l, dst_l, ts_l, label_l, batch_size, lr_model, tgan, num_layer, debug=False, partial_name=''):
    pred_prob = np.zeros(len(src_l))
    loss = 0
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        lr_model.eval()
        tgan.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance - 1, s_idx + batch_size)
            src_l_cut = src_l[s_idx:e_idx]
            dst_l_cut = dst_l[s_idx:e_idx]
            ts_l_cut = ts_l[s_idx:e_idx]
            label_l_cut = label_l[s_idx:e_idx]
            size = len(src_l_cut)
            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            loss += lr_criterion_eval(lr_prob, src_label).item()
            pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()

    auc_roc = roc_auc_score(label_l, pred_prob)

    # for debugging purpose; saves the true and predicted labels to file for further analysis
    if debug:
        name_info = partial_name.split(';')  # e.g. "wikipedia;tr;0"
        lbl_df = pd.DataFrame({'y_true': label_l, 'y_pred_prob': pred_prob})
        lbl_df.to_csv('./data/{}/TGAT_preds/{}_TGAT_{}_{}_pred.csv'.format(name_info[0], name_info[0],
                                                                           name_info[1], name_info[2]), index=False)

    return auc_roc, loss / num_instance


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='batch_size')  # previously, default=30
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=10, help='number of neighbors to sample')  # previously default=50
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
# parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')  # default=None
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')  # default=None
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information',
                    default='time')
parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--use_validation', action='store_true', help='Whether to use a validation set')
parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation set.')
parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of test set.')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
# NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
full_data, n_feat, e_feat, train_data, val_data, test_data = \
    get_data_node_clf(DATA, args.val_ratio, args.test_ratio, use_validation=args.use_validation)

max_idx = max(full_data.unique_nodes)

train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

full_ngh_finder = get_neighbor_finder(full_data, uniform=UNIFORM, max_node_idx=max_idx)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

for run_idx in range(args.n_runs):
    ### Model initialize
    tgan = TGAN(train_ngh_finder, n_feat, e_feat,
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
    # optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
    # criterion = torch.nn.BCELoss()
    tgan = tgan.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    logger.debug('num of training instances: {}'.format(num_instance))
    logger.debug('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list)

    logger.info('Loading saved TGAN model')
    model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
    tgan.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    tgan.eval()
    logger.info('TGAN models loaded')
    logger.info('Start training node classification task.')

    lr_model = LR(n_feat.shape[1])
    lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
    lr_model = lr_model.to(device)
    tgan.ngh_finder = full_ngh_finder
    idx_list = np.arange(len(train_data.sources))
    lr_criterion = torch.nn.BCELoss()
    lr_criterion_eval = torch.nn.BCELoss()

    for epoch in range(args.n_epoch):
        start_epoch = time.time()
        lr_pred_prob = np.zeros(len(train_data.sources))
        np.random.shuffle(idx_list)
        tgan = tgan.eval()
        lr_model = lr_model.train()
        # num_batch
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = train_data.sources[s_idx:e_idx]
            dst_l_cut = train_data.destinations[s_idx:e_idx]
            ts_l_cut = train_data.timestamps[s_idx:e_idx]
            label_l_cut = train_data.timestamps[s_idx:e_idx]

            size = len(src_l_cut)

            lr_optimizer.zero_grad()
            with torch.no_grad():
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)

            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            lr_loss = lr_criterion(lr_prob, src_label)
            lr_loss.backward()
            lr_optimizer.step()

        train_auc, train_loss = eval_epoch(train_data.sources, train_data.destinations, train_data.timestamps,
                                           train_data.labels, BATCH_SIZE, lr_model, tgan, num_layer=NODE_LAYER)
        # test_auc, test_loss = eval_epoch(test_data.sources, test_data.destinations, test_data.timestamps,
        #                                  test_data.labels, BATCH_SIZE, lr_model, tgan)
        val_auc, val_loss = eval_epoch(val_data.sources, val_data.destinations, val_data.timestamps,
                                       val_data.labels, BATCH_SIZE, lr_model, tgan, num_layer=NODE_LAYER)
        # torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
        logger.info(f'Epoch: {epoch}, train auc: {train_auc}, val auc: {val_auc}, time: {time.time() - start_epoch}')

    logger.info('>>> One forward pass to generate label prediction for training and test set (for debugging).')
    one_pass_start_time = time.time()
    eval_epoch(train_data.sources, train_data.destinations, train_data.timestamps,
               train_data.labels, BATCH_SIZE, lr_model, tgan, num_layer=NODE_LAYER, debug=True,
               partial_name=f"{DATA};tr;{run_idx}")
    logger.info('>>> Forward pass for run {} - training, took {} seconds.'.format(run_idx, time.time() - one_pass_start_time))
    one_pass_start_time = time.time()
    test_auc, test_loss = eval_epoch(test_data.sources, test_data.destinations, test_data.timestamps,
                                     test_data.labels, BATCH_SIZE, lr_model, tgan, num_layer=NODE_LAYER, debug=True,
                                     partial_name=f"{DATA};ts;{run_idx}")
    # torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
    logger.info(f'Run: {run_idx}, test auc: {test_auc}')
    logger.info('>>> Forward pass for run {} - test, took {} seconds.'.format(run_idx, time.time() - one_pass_start_time))
