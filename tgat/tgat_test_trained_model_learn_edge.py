"""
load a trained model for link prediction and test its performance on a set of positive and negative edges

Date: Jan. 24, 2022
"""

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
from utils import RandEdgeSampler, eval_one_epoch_modified, RandEdgeSampler_adversarial
from data_processing import get_data_link_pred

def args_parser():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
    parser.add_argument('--prefix', type=str, default='TGAT', help='prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=32, help='number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
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

    return args


def set_logger(args):
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
    return logger


def main():
    """
    main procedure for loading and testing a saved model
    """
    args = args_parser()

    NUM_NEIGHBORS = args.n_degree
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    USE_TIME = args.time
    AGG_METHOD = args.agg_method
    ATTN_MODE = args.attn_mode
    SEQ_LEN = NUM_NEIGHBORS
    DATA = args.data
    NUM_LAYER = args.n_layer
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    NEG_SAMPLE = args.neg_sample
    N_RUNS = args.n_runs
    PREFIX = args.prefix

    # setup logger
    LOG_FILENAME = './log/neg_sample_nre/{}_{}_{}_self_sup.log'.format(PREFIX, NEG_SAMPLE, DATA)
    logger = set_logger(LOG_FILENAME)
    logger.info("*** Arguments ***")
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
    if NEG_SAMPLE != 'rnd':
        logger.info("Negative Edge Sampling: {}".format(NEG_SAMPLE))
        # train_rand_sampler = RandEdgeSampler_NRE(train_data.sources, train_data.destinations, train_data.timestamps)
        # val_rand_sampler = RandEdgeSampler_NRE(full_data.sources, full_data.destinations, full_data.timestamps, seed=0)
        # nn_val_rand_sampler = RandEdgeSampler_NRE(new_node_val_data.sources, new_node_val_data.destinations,
        #                                           new_node_val_data.timestamps, seed=1)
        test_rand_sampler = RandEdgeSampler_adversarial(full_data.sources, full_data.destinations, full_data.timestamps,
                                                val_data.timestamps[-1], NEG_SAMPLE, seed=2)
        nn_test_rand_sampler = RandEdgeSampler_adversarial(new_node_test_data.sources,
                                                           new_node_test_data.destinations,
                                                           new_node_test_data.timestamps, val_data.timestamps[-1],
                                                           NEG_SAMPLE, seed=3)
    else:
        # train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
        # val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
        # nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
        test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
        nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                               new_node_test_data.destinations,
                                               seed=3)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    logger.info("************************************")
    logger.info("*********** Test starts *************")
    start_test = time.time()

    for i in range(N_RUNS):
        run_start_time = time.time()
        logger.info("************************************")
        logger.info("********** Run {} starts. **********".format(i))
        MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{i}.pth'

        ### Model initialize
        tgan = TGAN(train_ngh_finder, node_features, edge_features,
                    num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                    seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)

        # load saved parameters of the model
        tgan.load_state_dict(torch.load(MODEL_SAVE_PATH))
        tgan = tgan.to(device)
        tgan.eval()

        # testing phase use all information
        tgan.ngh_finder = full_ngh_finder

        # transductive (in terms of nodes)
        test_ap, test_auc_roc, test_avg_measures_dict = \
            eval_one_epoch_modified('test for old nodes', tgan, test_rand_sampler, test_data.sources,
                                    test_data.destinations, test_data.timestamps, test_data.labels, NUM_NEIGHBORS)

        # inductive (in terms of nodes)
        nn_test_ap, nn_test_auc_roc, nn_test_avg_measures_dict = \
            eval_one_epoch_modified('test for new nodes', tgan, nn_test_rand_sampler, new_node_test_data.sources,
                                    new_node_test_data.destinations, new_node_test_data.timestamps,
                                    new_node_test_data.labels, NUM_NEIGHBORS)

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

    logger.info('Info: Total elapsed time: {} seconds.'.format(time.time() - start_test))



if __name__ == '__main__':
    main()
