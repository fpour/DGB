"""
load a trained model for link prediction and test its performance on a set of positive and negative edges
"""

import math
import logging
import time
import sys
import argparse

import random
import torch
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from evaluation.evaluation import eval_edge_prediction_modified
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, RandEdgeSampler_adversarial
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)


def args_parser():
    parser = argparse.ArgumentParser('Self-Supervised Task - Test Phase Only.')
    # TGN model parameters
    parser.add_argument('--use_memory', action='store_true', help='Whether to augment the model with a node memory')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
        "gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each user')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--dyrep', action='store_true',
                        help='Whether to run the dyrep model')

    # Data related parameters
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                        default='wikipedia')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')

    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation set.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of test set.')
    parser.add_argument('--neg_sample', type=str, default='rnd', help='Strategy for the edge negative sampling.')

    # Parameters for loading the model
    parser.add_argument('--model', type=str, default='tgn', help='The name of the model to load.')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args


def set_logger(log_filename):
    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    """
    main procedure for loading and testing a saved model
    """
    args = args_parser()

    if args.model == 'tgn':
        prefix = 'tgn_attn'
        log_prefix = 'TGN'
    elif args.model == 'jodie':
        prefix = 'jodie_rnn'
        log_prefix = 'JODIE'
    elif args.model == 'dyrep':
        prefix = 'dyrep_rnn'
        log_prefix = 'DyRep'
    else:
        print('Not a valid model name!')
        prefix = ''
        log_prefix = ''
        exit(-1)

    USE_MEMORY = args.use_memory
    DATA = args.data
    GPU = args.gpu
    NUM_NEIGHBORS = args.n_degree
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    NUM_LAYER = args.n_layer
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    NEG_SAMPLE = args.neg_sample
    N_RUNS = args.n_runs

    # setup logger
    LOG_FILENAME = './log/neg_sample/{}_{}_{}_self_sup.log'.format(log_prefix, NEG_SAMPLE, DATA)
    logger = set_logger(LOG_FILENAME)
    logger.info("*** Arguments ***")
    logger.info(args)

    ### Extract data for training, validation and testing
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_data(DATA, args.val_ratio, args.test_ratio,
                                  different_new_nodes_between_val_and_test=args.different_new_nodes,
                                  randomize_features=args.randomize_features)

    # Initialize training neighbor finder to retrieve temporal graph
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers.
    # Set seeds for validation and testing so negatives are the same across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    if NEG_SAMPLE != 'rnd':  # adversarial_sampling
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
        # nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
        #                                       seed=1)
        test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
        nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                               new_node_test_data.destinations,
                                               seed=3)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    logger.info("************************************")
    logger.info("*********** Test starts *************")
    start_test = time.time()

    for i_run in range(N_RUNS):
        start_run = time.time()
        logger.info("************************************")
        logger.info("*********** Run {} starts *************".format(i_run))

        MODEL_SAVE_PATH = f'./saved_models/{prefix}-{args.data}-{i_run}.pth'

        # Initialize Model
        tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                  edge_features=edge_features, device=device,
                  n_layers=NUM_LAYER,
                  n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                  message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater,
                  n_neighbors=NUM_NEIGHBORS,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  dyrep=args.dyrep)
        # load saved parameters of the model
        tgn.load_state_dict(torch.load(MODEL_SAVE_PATH))
        tgn = tgn.to(device)
        tgn.eval()

        # Training has finished, we have loaded the best model, and we want to backup its current
        # memory (which has seen validation edges) so that it can also be used when testing on unseen
        # nodes
        if USE_MEMORY:
            val_memory_backup = tgn.memory.backup_memory()

        tgn.embedding_module.neighbor_finder = full_ngh_finder

        # transductive task
        test_ap, test_auc, test_measures_dict = eval_edge_prediction_modified(model=tgn,
                                                                     negative_edge_sampler=test_rand_sampler,
                                                                     data=test_data,
                                                                     n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
            tgn.memory.restore_memory(val_memory_backup)

        # Inductive task: Test on unseen nodes
        nn_test_ap, nn_test_auc, nn_test_measures_dict = eval_edge_prediction_modified(model=tgn,
                                                                              negative_edge_sampler=nn_test_rand_sampler,
                                                                              data=new_node_test_data,
                                                                              n_neighbors=NUM_NEIGHBORS)

        logger.info('Performance of the {} model for the test set.'.format(args.model))
        logger.info('Network Name: {}, Model: {}'.format(DATA, args.model))
        logger.info('Test statistics: *** Old Nodes (Transductive) ***')
        logger.info('Test statistics: Old nodes -- auc_inherent: {}'.format(test_auc))
        logger.info('Test statistics: Old nodes -- ap_inherent: {}'.format(test_ap))
        logger.info('Test statistics: New nodes -- auc_inherent: {}'.format(nn_test_auc))
        logger.info('Test statistics: New nodes -- ap_inherent: {}'.format(nn_test_ap))

        # extra performance measures
        # Note: just prints out for the Test set!
        for measure_name, measure_value in test_measures_dict.items():
            logger.info('Test statistics: Old nodes -- {}: {}'.format(measure_name, measure_value))
        for measure_name, measure_value in nn_test_measures_dict.items():
            logger.info('Test statistics: New nodes -- {}: {}'.format(measure_name, measure_value))

        logger.info('Info: Run {} elapsed time: {} seconds.'.format(i_run, (time.time() - start_run)))

    logger.info('Info: Total elapsed time: {} seconds.'.format(time.time() - start_test))


if __name__ == '__main__':
    main()
