"""
This is the exact copy of TGN/utils/preprocess_data.py
Just to preprocess and save edge lists as it is required for CAW-N
"""
"""
Goal: to convert the initial edge list of the network to the appropriate format needed by the model
Note the "pre" in the name of the file: preprocess_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)  # skip the first line
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])  # user_id
      i = int(e[1])  # item_id

      ts = float(e[2])  # timestamp  --> assumed in ascending order (I've checked it)
      label = float(e[3])  # int(e[3])  # state_label

      feat = np.array([float(x) for x in e[4:]])  # edge features

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


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True, n_node_feat=172):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, n_node_feat))

  new_df.to_csv(OUT_DF)  # edge-list
  np.save(OUT_FEAT, feat)   # edge features
  np.save(OUT_NODE_FEAT, rand_feat)  # node features


###########################################################################
### scripts
###########################################################################

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
parser.add_argument('--n_node_feat', type=int, default=172, help='Number of node initial features')

args = parser.parse_args()

if args.data != 'reddit' and args.data != 'wikipedia':
  run(args.data, bipartite=False, n_node_feat=args.n_node_feat)
else:
  run(args.data, bipartite=args.bipartite, n_node_feat=args.n_node_feat)