import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import sys


def get_one_hot(valid_len, tot_len):
    return np.concatenate((np.eye(valid_len), np.zeros((valid_len, tot_len - valid_len))), axis=-1)


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        print(s)
        # for idx, line in tqdm(enumerate(f)):
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)


def reindex(df, jodie_data):
    if jodie_data:
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df = df.copy()
        new_df.i = new_i

        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df = df.copy()
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    return new_df


def run(args):
    data_name = args.dataset
    node_edge_feat_dim = args.node_edge_feat_dim
    PATH = './data/{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    jodie_data = data_name in ['wikipedia', 'reddit', 'mooc', 'lastfm']
    print('preprocess {} dataset ...'.format(data_name))
    df, feat = preprocess(PATH)
    new_df = reindex(df, jodie_data)

    if not args.one_hot_node:
        empty = np.zeros(feat.shape[1])[np.newaxis, :]
        feat = np.vstack([empty, feat])
        max_idx = max(new_df.u.max(), new_df.i.max())
        rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
        if 'socialevolve' in data_name:
            feat = np.zeros((feat.shape[0], node_edge_feat_dim))
            rand_feat = np.zeros((rand_feat.shape[0], node_edge_feat_dim))
        if data_name in ['LegisEdgelist', 'UNvote', 'UNtrade', 'canVote', 'UNtrade', 'covid_20200301_20200630']:  # refine to
            # make no problems for number of edge/node features
            feat_reshaped = np.reshape(feat, (feat.shape[0], 1))
            feat_zero = np.zeros((feat.shape[0], node_edge_feat_dim-1))
            feat = np.hstack((feat_reshaped, feat_zero))
            rand_feat = np.zeros((rand_feat.shape[0], node_edge_feat_dim))
        print('node feature shape:', rand_feat.shape)
        print('edge feature shape:', feat.shape)
    else:
        # (obsolete branch) TODO: still problematic, add one-hot encoding if possible
        empty = np.zeros(feat.shape[1])[np.newaxis, :]
        feat = np.vstack([empty, feat])
        feat = np.concatenate()
        max_idx = max(new_df.u.max(), new_df.i.max())
        rand_feat = get_one_hot(max_idx + 1, feat.shape[1])

        print('one-hot node feature:', rand_feat.shape)
    print(feat.shape)
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)


parser = argparse.ArgumentParser('Interface for propressing csv source data for CAW framework')
parser.add_argument('--dataset',
                    choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'socialevolve', 'socialevolve_1month', 'socialevolve_2weeks',
                             'LegisEdgelist', 'UNvote','UNtrade' , 'canVote', 'covid_20200301_20200630'],
                    help='specify one dataset to preprocess')
parser.add_argument('--node_edge_feat_dim', default=172,
                    help='number of dimensions for 0-padded node and edge features')
parser.add_argument('--one-hot-node', type=bool, default=False,
                    help='using one hot embedding for node (which means inductive learning is impossible though)')
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

run(args)
