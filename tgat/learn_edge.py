"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
# import numba

from sklearn.metrics import *

from module import TGAN
from graph import get_neighbor_finder
from utils import EarlyStopMonitor, RandEdgeSampler, eval_one_epoch_original
from data_processing import get_data_link_pred

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=32, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=64, help='Dimension of the node embedding')
parser.add_argument('--time_dim', type=int, default=64, help='Dimension of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod',
                    help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information',
                    default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation set.')
parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of test set.')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--neg_sample', type=str, default='rnd', help='Strategy for the edge negative sampling.')

try:
    args = parser.parse_args()
    print("ARGS:\n{}".format(args))
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
# NEW_NODE = args.new_node  # no usage!
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

# MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

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
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
    get_data_link_pred(DATA, val_pointer=args.val_ratio, test_pointer=args.test_ratio,
                       different_new_nodes_between_val_and_test=args.different_new_nodes,
                       randomize_features=args.randomize_features)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers.
# Set seeds for validation and testing so negatives are the same across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes

logger.info('Negative Sampling Method: >> Random <<.')
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

for i in range(args.n_runs):
    run_start_time = time.time()
    logger.info("************************************")
    logger.info("********** Run {} starts. **********".format(i))

    ### Model initialize
    tgan = TGAN(train_ngh_finder, node_features, edge_features,
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
    optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    tgan = tgan.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list)

    early_stopper = EarlyStopMonitor()
    for epoch in range(NUM_EPOCH):
        epoch_start_time = time.time()
        # Training
        # training use only training graph
        tgan.ngh_finder = train_ngh_finder
        acc, ap, auc_roc, m_loss = [], [], [], []
        np.random.shuffle(idx_list)
        logger.info('start epoch number {}'.format(epoch))
        for k in range(num_batch):
            # percent = 100 * k / num_batch
            # if k % int(0.2 * num_batch) == 0:
            #     logger.info('progress: {0:10.4f}'.format(percent))

            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut, dst_l_cut = train_data.sources[s_idx:e_idx], train_data.destinations[s_idx:e_idx]
            ts_l_cut = train_data.timestamps[s_idx:e_idx]
            label_l_cut = train_data.labels[s_idx:e_idx]
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)

            optimizer.zero_grad()
            tgan = tgan.train()
            pos_prob, neg_prob = tgan.contrast_original(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            loss = criterion(pos_prob, pos_label)
            loss += criterion(neg_prob, neg_label)

            loss.backward()
            optimizer.step()
            # get training results
            with torch.no_grad():
                tgan = tgan.eval()
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                pred_label = pred_score > 0.5  # for computing accuracy; datasets are mostly very imbalance!
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))

                m_loss.append(loss.item())
                auc_roc.append(roc_auc_score(true_label, pred_score))

        # validation phase use all information
        tgan.ngh_finder = full_ngh_finder
        val_ap, val_auc_roc, val_avg_measures_dict = \
            eval_one_epoch_original('val for old nodes', tgan, val_rand_sampler, val_data.sources,
                                    val_data.destinations, val_data.timestamps, val_data.labels, NUM_NEIGHBORS)

        if DATA != 'uci':
            nn_val_ap, nn_val_auc_roc, nn_val_avg_measures_dict = \
                eval_one_epoch_original('val for new nodes', tgan, val_rand_sampler, new_node_val_data.sources,
                                        new_node_val_data.destinations, new_node_val_data.timestamps, new_node_val_data.labels, NUM_NEIGHBORS)

        logger.info('epoch: {}:'.format(epoch))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        # logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc_roc), val_auc_roc, nn_val_auc_roc))
        # logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))
        epoch_time = time.time() - epoch_start_time
        logger.info('epoch {} took {:.2f}s.'.format(epoch, epoch_time))

        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgan.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgan.eval()
            break
        else:
            torch.save(tgan.state_dict(), get_checkpoint_path(epoch))

    # testing phase use all information
    tgan.ngh_finder = full_ngh_finder
    test_ap, test_auc_roc, test_avg_measures_dict = \
        eval_one_epoch_original('test for old nodes', tgan, test_rand_sampler, test_data.sources,
                                test_data.destinations, test_data.timestamps, test_data.labels, NUM_NEIGHBORS)

    nn_test_ap, nn_test_auc_roc, nn_test_avg_measures_dict = \
        eval_one_epoch_original('test for new nodes', tgan, nn_test_rand_sampler, new_node_test_data.sources,
                                new_node_test_data.destinations, new_node_test_data.timestamps, new_node_test_data.labels, NUM_NEIGHBORS)

    logger.info(
        'Test statistics: Old nodes -- auc_inherent: {}'.format(test_auc_roc))
    logger.info(
        'Test statistics: Old nodes -- ap_inherent: {}'.format(test_ap))
    logger.info(
        'Test statistics: New nodes -- auc_inherent: {}'.format(nn_test_auc_roc))
    logger.info(
        'Test statistics: New nodes -- ap_inherent: {}'.format(nn_test_ap))

    # extra performance measures
    # Note: just prints out for the Test set!
    for measure_name, measure_value in test_avg_measures_dict.items():
        logger.info('Test statistics: Old nodes -- {}: {}'.format(measure_name, measure_value))
    for measure_name, measure_value in nn_test_avg_measures_dict.items():
        logger.info('Test statistics: New nodes -- {}: {}'.format(measure_name, measure_value))

    per_run_time = time.time() - run_start_time
    logger.info('Run {} took {:.2f}s.'.format(i, per_run_time))

    logger.info('Saving TGAN model')
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{i}.pth'
    torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGAN models saved')
