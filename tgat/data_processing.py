"""
Similar to the file with the same name in the TGN
Goal: to generate full, train, validation, and test split of the data in appropriate format used by TGAT model
Date: 09/14/2021
"""
import numpy as np
import random
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data_link_pred(dataset_name, val_pointer, test_pointer, different_new_nodes_between_val_and_test=False,
             randomize_features=False):
    """
    The main function to generate data splits for link prediction task (inductive & transductive settings)
    """
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
    node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

    # additional for CAW data specifically
    if dataset_name in ['enron', 'socialevolve', 'uci', 'copenhagen']:
        node_zero_padding = np.zeros((node_features.shape[0], 172 - node_features.shape[1]))
        node_features = np.concatenate([node_features, node_zero_padding], axis=1)
        edge_zero_padding = np.zeros((edge_features.shape[0], 172 - edge_features.shape[1]))
        edge_features = np.concatenate([edge_features, edge_zero_padding], axis=1)

    if randomize_features:
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

    if val_pointer < 1:  # the ratio of the validation and test set have been given
        val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_pointer - test_pointer), (1 - test_pointer)]))
    else:  # the timestamp from which validation and test set start is given
        val_time = val_pointer
        test_time = test_pointer

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(2020)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps > val_time]).union(
        set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all
    # their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))  # the same 10% of CAW-N?!

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set  # new nodes that are not in the training set

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    if different_new_nodes_between_val_and_test:  # 'new_test_node_set' is used
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array(
            [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array(
            [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)

    else:  # 'new_node_set' is used
        edge_contains_new_node_mask = np.array(
            [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], labels[new_node_val_mask])

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
        len(new_test_node_set)))

    return node_features, edge_features, full_data, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data


def get_data_node_clf(dataset_name, val_pointer, test_pointer, use_validation=False):
    """
    generate data splits for node classification
    """
    ### Load data and train val test split
    g_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    e_feat = np.load('./data/ml_{}.npy'.format(dataset_name))
    n_feat = np.load('./data/ml_{}_node.npy'.format(dataset_name))

    if val_pointer < 1:  # the ratio of the validation and test set have been given
        val_time, test_time = list(np.quantile(g_df.ts, [(1 - val_pointer - test_pointer), (1 - test_pointer)]))
    else:  # the timestamp from which validation and test set start is given
        val_time = val_pointer
        test_time = test_pointer

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values

    random.seed(2020)

    train_mask = ts_l <= val_time if use_validation else ts_l <= test_time
    test_mask = ts_l > test_time
    val_mask = np.logical_and(ts_l <= test_time, ts_l > val_time) if use_validation else test_mask

    full_data = Data(src_l, dst_l, ts_l, e_idx_l, label_l)
    train_data = Data(src_l[train_mask], dst_l[train_mask], ts_l[train_mask],
                      e_idx_l[train_mask], label_l[train_mask])
    val_data = Data(src_l[val_mask], dst_l[val_mask], ts_l[val_mask],
                    e_idx_l[val_mask], label_l[val_mask])
    test_data = Data(src_l[test_mask], dst_l[test_mask], ts_l[test_mask],
                     e_idx_l[test_mask], label_l[test_mask])

    return full_data, n_feat, e_feat, train_data, val_data, test_data