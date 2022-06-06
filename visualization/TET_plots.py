"""
TET Plots
"""

import math
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker

# some parameters to be used for drawing
E_ABSENT = 0
E_PRESENCE_GENERAL = 1
E_SEEN_IN_TRAIN = 2
E_IN_TEST = 3
E_NOT_IN_TEST = 4

TEST_RATIO = 0.15

# new color controlling parameters; Date: Dec. 22, 2021
E_ONLY_TRAIN = 10
E_TRAIN_AND_TEST = 20
E_TRANSDUCTIVE = 30
E_INDUCTIVE = 40


def generate_edge_per_ts_and_idx_map(edgelist_df, directed):
    # edges per each timestamps & (source, destination) to edge index
    print("Info: Network Directed: {}".format(directed))
    print("Info: Total Number of edges: {}".format(edgelist_df.shape[0]))
    edges_per_ts = {}
    for idx, row in tqdm(edgelist_df.iterrows()):
        if directed:
            # 1. edges per timestamp
            if row['t'] not in edges_per_ts:
                edges_per_ts[row['t']] = {}
                edges_per_ts[row['t']][(row['u'], row['v'])] = 1
            else:
                if (row['u'], row['v']) not in edges_per_ts[row['t']]:
                    edges_per_ts[row['t']][(row['u'], row['v'])] = 1
        else:
            print("Not implemented yet!")
            exit(1)

    edge_last_ts = generate_edge_last_timestamp(edges_per_ts)
    edge_idx_map = generate_edge_idx_map(edges_per_ts, edge_last_ts)
    idx_edge_map = {v: k for k, v in edge_idx_map.items()}  # key: edge index; value: actual edge (source, destination)

    print("Info: Number of distinct edges (from index-edge map): {}".format(len(idx_edge_map)))

    return edges_per_ts, idx_edge_map, edge_idx_map


def generate_edge_last_timestamp(edges_per_ts):
    """generates a dictionary containing the last timestamp of each edge"""
    edge_last_ts = {}
    for ts, e_list in edges_per_ts.items():
        for e in e_list:
            if e not in edge_last_ts:
                edge_last_ts[e] = ts
            else:
                edge_last_ts[e] = max(ts, edge_last_ts[e])
    return edge_last_ts


def generate_edge_idx_map(edges_per_ts, edge_last_ts):
    """
    generates index for edges according to two-level sorting policy:
    1. the first level is based on their first appearance timestamp
    2. the second level is based on their last appearance timestamp
    """
    edge_idx_map = {}  # key: actual edge (source, destination), value: edge index
    distinct_edge_idx = 0
    for ts, ts_e_list in edges_per_ts.items():
        e_last_ts_this_timestamp = {}
        for e in ts_e_list:
            e_last_ts_this_timestamp[e] = edge_last_ts[e]
        e_last_ts_this_timestamp = dict(sorted(e_last_ts_this_timestamp.items(), key=lambda item: item[1]))
        for e in e_last_ts_this_timestamp:
            if e not in edge_idx_map:
                edge_idx_map[e] = distinct_edge_idx
                distinct_edge_idx += 1

    return edge_idx_map


def generate_edge_presence_matrix(unique_ts_list, idx_edge_map, edge_idx_map, edges_per_ts):
    num_unique_ts = len(unique_ts_list)
    num_unique_edge = len(idx_edge_map)
    e_presence_mat = np.zeros([num_unique_ts, num_unique_edge], dtype=np.int8)
    unique_ts_list = np.sort(unique_ts_list)

    for x, ts in tqdm(enumerate(unique_ts_list)):
        es_ts = edges_per_ts[ts]
        for e in es_ts:
            e_presence_mat[num_unique_ts - x - 1, edge_idx_map[e]] = E_PRESENCE_GENERAL

    return e_presence_mat


class Fig_Param:
    def __init__(self, network_name, fig_name, figsize, axis_title_font_size, x_font_size, y_font_size, axis_tick_gap,
                 timestamp_split_cross_mark_offset):
        self.network_name = network_name
        self.fig_name = fig_name
        self.figsize = figsize
        self.axis_title_font_size = axis_title_font_size
        self.x_font_size = x_font_size
        self.y_font_size = y_font_size
        self.axis_tick_gap = axis_tick_gap
        self.timestamp_split_cross_mark_offset = timestamp_split_cross_mark_offset


def plot_edge_presence_matrix(e_presence_mat, test_split_ts_value, unique_ts_list,
                              idx_edge_list, fig_param, add_frames=False):
    print("Info: plotting edge presence heatmap for {} ...".format(fig_param.fig_name))

    fig, ax = plt.subplots(figsize=fig_param.figsize)
    plt.subplots_adjust(bottom=0.3, left=0.2)

    # colors = ['white',  # E_ABSENCE
    #           '#67a9cf',  # E_ONLY_TRAIN
    #           '#ef8a62',  # E_TRAIN_AND_TEST
    #           '#ef8a62',  # E_TRANSDUCTIVE
    #           '#b2182b'  # E_INDUCTIVE
    #           ]
    colors = ['white',  # E_ABSENCE
              '#018571',  # E_ONLY_TRAIN    2c7bb6
              '#fc8d59',  # E_TRAIN_AND_TEST
              '#fc8d59',  # E_TRANSDUCTIVE
              '#b2182b'  # E_INDUCTIVE
              ]

    frame_color = "grey" # "#bababa"
    time_split_color = "black"
    axis_title_font_size = fig_param.axis_title_font_size
    x_font_size = fig_param.x_font_size
    y_font_size = fig_param.y_font_size

    ax = sns.heatmap(e_presence_mat, cmap=sns.color_palette(colors, as_cmap=True), cbar=False)

    # processing x-axis
    x_gaps = np.linspace(0, len((idx_edge_list)), num=5)
    x_labels = x_gaps / len(idx_edge_list)
    x_labels = [int(100*x) for x in x_labels]
    plt.xticks(x_gaps, x_labels, rotation=0, fontsize=x_font_size)

    # processing y-axis
    t_gaps = np.linspace(0, len(unique_ts_list), num=5)
    t_labels = [int(len(unique_ts_list) - tidx) for tidx in t_gaps]
    plt.yticks(t_gaps, t_labels, rotation=90, fontsize=y_font_size)

    # axis & title
    # plt.margins(x=0)
    plt.xlabel("Percentage of observed edges", fontsize=axis_title_font_size)
    plt.ylabel("Timestamp", fontsize=axis_title_font_size)

    # requirements for additional features
    x_length = e_presence_mat.shape[1] - 1
    y_length = e_presence_mat.shape[0] - 1
    test_split_idx_value = y_length - test_split_ts_value
    e_border_idx = 0
    for e_idx in range(e_presence_mat.shape[1] - 1, -1, -1):
        if e_presence_mat[y_length - test_split_ts_value, e_idx] != E_ABSENT:
            e_border_idx = e_idx
            break

    # rectangle for different parts of the dataset
    if add_frames:
        print("Info: Border edge index:", e_border_idx)
        print("Info: Test split timestamp value:", test_split_ts_value)
        rect_train = plt.Rectangle((0, y_length - test_split_ts_value + 0.085), e_border_idx, test_split_ts_value + 0.9,
                                   fill=False, linewidth=2, edgecolor=frame_color)
        rect_test_mayseen = plt.Rectangle((0, 0), e_border_idx, y_length - test_split_ts_value - 0.1,
                                          fill=False, linewidth=2, edgecolor=frame_color)
        rect_test_new = plt.Rectangle((e_border_idx, 0), x_length - e_border_idx,
                                      y_length - test_split_ts_value - 0.1,
                                      fill=False, linewidth=2, edgecolor=frame_color)
        ax = ax or plt.gca()
        ax.add_patch(rect_train)
        ax.add_patch(rect_test_mayseen)
        ax.add_patch(rect_test_new)

    # test split horizontal line
    plt.axhline(y=test_split_idx_value, color=time_split_color, linestyle="--", linewidth=2, label='x')
    plt.text(x=0, y=test_split_idx_value, s='x', color=time_split_color, va='center', ha='center',
             fontsize=y_font_size, fontweight='heavy')

    if fig_param.fig_name != "":
        print("Info: file name: {}".format(fig_param.fig_name))
        plt.savefig(fig_param.fig_name)
        plt.close()

    plt.show()
    print("Info: plotting done!")


def load_continuous_edgelist(fname, interval_size=86400):
    """
    load temporal edgelist into a dictionary
    assumption: the edges are ordered in increasing order of their timestamp
    '''
    the timestamp in the edgelist is based cardinal
    more detail see here: https://github.com/srijankr/jodie
    need to merge edges in a period of time into an interval
    86400 is # of secs in a day, good interval size
    '''
    # suitable for "Wikipedia" and "Reddit"
    """
    u_idx = 1
    v_idx = 2
    ts_idx = 3

    edges_per_ts, edge_idx_map = {}, {}
    total_n_edges = 0
    distinct_edge_idx = 0
    with open(fname) as f:
        s = next(f)  # skip the first line
        for idx, line in enumerate(f):
            total_n_edges += 1
            e = line.strip().split(',')
            u = e[u_idx]  # source node
            v = e[v_idx]  # destination node
            ts = float(e[ts_idx])  # timestamp
            ts_bin_id = int(ts / interval_size)
            # 1. edges_per_ts
            if ts_bin_id not in edges_per_ts:
                edges_per_ts[ts_bin_id] = {}
                edges_per_ts[ts_bin_id][(u, v)] = 1
            else:
                if (u, v) not in edges_per_ts[ts_bin_id]:
                    edges_per_ts[ts_bin_id][(u, v)] = 1

    edge_last_ts = generate_edge_last_timestamp(edges_per_ts)
    edge_idx_map = generate_edge_idx_map(edges_per_ts, edge_last_ts)
    idx_edge_map = {v: k for k, v in edge_idx_map.items()}  # key: edge index; value: actual edge (source, destination)

    print("Info: Loading edge-list: Maximum timestamp is ", ts)
    print("Info: Loading edge-list: Maximum timestamp-bin-id is", ts_bin_id)
    print("Info: Loading edge-list: Total number of edges:", total_n_edges)
    print("Info: Loading edge-list: Number of distinct edges:", len(idx_edge_map))
    return edges_per_ts, idx_edge_map, edge_idx_map


def load_retweet_edgelist(fname):
    """
    treat each year as a timestamp
    """
    edgelist = open(fname, "r")
    edgelist.readline()
    lines = list(edgelist.readlines())
    edgelist.close()

    edges_per_ts, edge_idx_map = {}, {}
    total_n_edges = 0
    distinct_edge_idx = 0

    for i in range(1, len(lines)):
        total_n_edges += 1
        line = lines[i]
        values = line.split(',')
        t = values[0]
        u = values[1]
        v = values[2].strip('\n')
        # edges_per_ts
        if t not in edges_per_ts:
            edges_per_ts[t] = {}
            edges_per_ts[t][(u, v)] = 1
        else:
            if (u, v) not in edges_per_ts[t]:
                edges_per_ts[t][(u, v)] = 1

    edge_last_ts = generate_edge_last_timestamp(edges_per_ts)
    edge_idx_map = generate_edge_idx_map(edges_per_ts, edge_last_ts)
    idx_edge_map = {v: k for k, v in edge_idx_map.items()}  # key: edge index; value: actual edge (source, destination)

    print("Info: Loading tweeter: Number of loaded edges: " + str(total_n_edges))
    print("Info: Loading tweeter: Available timestamps:", edges_per_ts.keys())
    print("Info: Loading tweeter: Number of distinct edges:", len(idx_edge_map))
    return edges_per_ts, idx_edge_map, edge_idx_map


def load_flight_edgelist(fname):
    """
    treat each year as a timestamp
    """
    edgelist = open(fname, "r")
    edgelist.readline()
    lines = list(edgelist.readlines())
    edgelist.close()

    edges_per_ts, edge_idx_map = {}, {}
    total_n_edges = 0
    distinct_edge_idx = 0

    for i in range(1, len(lines)):
        total_n_edges += 1
        line = lines[i]
        values = line.split(',')
        t = values[-1].strip('\n')
        u = values[0]
        v = values[1]
        # edges_per_ts
        if t not in edges_per_ts:
            edges_per_ts[t] = {}
            edges_per_ts[t][(u, v)] = 1
        else:
            if (u, v) not in edges_per_ts[t]:
                edges_per_ts[t][(u, v)] = 1

    edge_last_ts = generate_edge_last_timestamp(edges_per_ts)
    edge_idx_map = generate_edge_idx_map(edges_per_ts, edge_last_ts)
    idx_edge_map = {v: k for k, v in edge_idx_map.items()}  # key: edge index; value: actual edge (source, destination)

    print("Info: Loading flights: Number of loaded edges: " + str(total_n_edges))
    print("Info: Loading flights: Available timestamps:", edges_per_ts.keys())
    print(("Info: Loading flights: Number of distinct edges:", len(edge_idx_map)))

    return edges_per_ts, idx_edge_map, edge_idx_map


def process_presence_matrix(e_presence_matrix, test_ratio_p):
    """
    there are 4 types of edge presence:
    1. only in train
    2. in train and in test
    3. in test and train (which is the number 2 but in later timestamps)
    4. only in test
    X: timestamp
    Y: edge index
    """
    num_unique_ts = e_presence_matrix.shape[0]
    num_unique_edges = e_presence_matrix.shape[1]
    ts_idx_list = [i for i in range(num_unique_ts)]
    test_split_ts_value = int(np.quantile(ts_idx_list, test_ratio_p))
    train_ts_list = [ts for ts in ts_idx_list if ts <= test_split_ts_value]  # any timestamp in train/validation split
    test_ts_list = [ts for ts in ts_idx_list if ts > test_split_ts_value]  # test_split_ts_value is in train
    # first level processing: differentiate train set edges: 1) Only in train set, 2) in train & test set
    for tr_ts in train_ts_list:
        for eidx in range(num_unique_edges):
            if e_presence_matrix[num_unique_ts - tr_ts - 1, eidx] == E_PRESENCE_GENERAL:
                for test_ts_idx in range(test_split_ts_value + 1, num_unique_ts):
                    if e_presence_matrix[num_unique_ts - test_ts_idx - 1, eidx] == E_PRESENCE_GENERAL:  # if seen in
                        # the test set
                        e_presence_matrix[num_unique_ts - tr_ts - 1, eidx] = E_TRAIN_AND_TEST
                        break
    # differentiate test set edges: 1) transductive (seen in train, repeating in test), 2) inductive (only in test)
    for ts in test_ts_list:
        for eidx in range(num_unique_edges):
            if e_presence_matrix[num_unique_ts - ts - 1, eidx] == E_PRESENCE_GENERAL:
                for prev_ts_idx in range(test_split_ts_value, -1, -1):
                    if e_presence_matrix[num_unique_ts - prev_ts_idx - 1, eidx] == E_TRAIN_AND_TEST:  # if seen in
                        # the training set
                        e_presence_matrix[num_unique_ts - ts - 1, eidx] = E_TRANSDUCTIVE
                        break
    # second level processing
    for ts in range(num_unique_ts):
        for eidx in range(num_unique_edges):
            if ts <= test_split_ts_value:
                if e_presence_matrix[num_unique_ts - ts - 1, eidx] == E_PRESENCE_GENERAL:
                    e_presence_matrix[num_unique_ts - ts - 1, eidx] = E_ONLY_TRAIN
            else:
                if e_presence_matrix[num_unique_ts - ts - 1, eidx] == E_PRESENCE_GENERAL:
                    e_presence_matrix[num_unique_ts - ts - 1, eidx] = E_INDUCTIVE

    return e_presence_matrix, test_split_ts_value


def from_temporal_edgelist_to_edge_presence_plot(edgelist_filename, network_name, fig_name,
                                                 add_frame=False, directed=True):
    """
    read a temporal edge-list and generate a matrix which specifies the existence of edges over time
    """
    if network_name in ['US Legislative', 'Canadian Vote', 'UN Trade', 'UN Vote']:
        # read edge-list and unify the column names
        edgelist_df = pd.read_csv(edgelist_filename)
        edgelist_df.columns = ['t', 'u', 'v', 'w']
        # unique edges
        grouped_edges = edgelist_df.groupby(['u', 'v'])
        num_unique_edges = len(grouped_edges)
        print("Info: Number of unique edges: {}".format(num_unique_edges))
        # unique timestamps
        unique_ts_list = list(set(edgelist_df['t'].tolist()))
        num_unique_ts = len(unique_ts_list)
        print("Info: Number of unique timestamps: {}".format(num_unique_ts))
        # generate edges per timestamp and index-edge map
        edges_per_ts, idx_edge_map, edge_idx_map = generate_edge_per_ts_and_idx_map(edgelist_df, directed)

    elif network_name in ['Reddit', 'Wikipedia', 'MOOC']:
        edges_per_ts, idx_edge_map, edge_idx_map = load_continuous_edgelist(edgelist_filename)
    elif network_name in ['LastFM', 'Enron']:
        interval_size = 86400 * 30
        edges_per_ts, idx_edge_map, edge_idx_map = load_continuous_edgelist(edgelist_filename, interval_size)
    elif network_name in [ 'UCI', 'Social Evo.']:
        interval_size = 86400 * 5
        edges_per_ts, idx_edge_map, edge_idx_map = load_continuous_edgelist(edgelist_filename, interval_size)
    elif network_name in ['Flights']:
        edges_per_ts, idx_edge_map, edge_idx_map = load_flight_edgelist(edgelist_filename)
    elif network_name in ['Political Science Retweet Network']:
        edges_per_ts, idx_edge_map, edge_idx_map = load_retweet_edgelist(edgelist_filename)
    else:
        print("Info: Invalid network name!")
        exit(-1)

    unique_ts_list = list(edges_per_ts.keys())
    e_presence_mat = generate_edge_presence_matrix(unique_ts_list, idx_edge_map, edge_idx_map, edges_per_ts)
    e_presence_mat, test_split_ts_value = process_presence_matrix(e_presence_mat, test_ratio_p=0.85)
    print("Info: edge-presence-matrix shape: {}".format(e_presence_mat.shape))

    fig_param = set_fig_param(network_name, fig_name)

    plot_edge_presence_matrix(e_presence_mat, test_split_ts_value, unique_ts_list, list(idx_edge_map.keys()),
                              fig_param, add_frames=add_frame)


def set_fig_param_v1(network_name, fig_name):
    # general settings
    coeff = 2  # 1.618
    y_size = 5.5
    x_size = lambda y_size : int(y_size * coeff)
    figsize = (x_size(y_size), y_size)
    axis_title_font_size = 20
    x_font_size = 18
    y_font_size = 18
    axis_tick_gap = 20
    timestamp_split_cross_mark_offset = 1

    if network_name in ['US Legislative', 'Canadian Vote', 'UN Trade', 'UN Vote']:
        axis_tick_gap = axis_tick_gap * 0.35

    elif network_name in ['Reddit', 'Wikipedia', 'UCI', 'Social Evo.', 'Flights', 'LastFM', 'MOOC']:
        axis_tick_gap = axis_tick_gap * 0.5

    elif network_name in ['Enron']:
        axis_tick_gap = axis_tick_gap * 0.4

    fig_param = Fig_Param(network_name, fig_name,
                          figsize, axis_title_font_size,
                          x_font_size,
                          y_font_size,
                          axis_tick_gap,
                          timestamp_split_cross_mark_offset)

    return fig_param


def set_fig_param(network_name, fig_name):
    # general settings
    figsize = (9, 5)
    axis_title_font_size = 20
    x_font_size = 22
    y_font_size = 22
    axis_tick_gap = 20
    timestamp_split_cross_mark_offset = 1

    if network_name in ['US Legislative', 'Canadian Vote', 'UN Trade', 'UN Vote']:
        axis_tick_gap = axis_tick_gap * 0.35

    elif network_name in ['Reddit', 'Wikipedia', 'UCI', 'Social Evo.', 'Flights', 'LastFM', 'MOOC']:
        axis_tick_gap = axis_tick_gap * 0.5

    elif network_name in ['Enron']:
        axis_tick_gap = axis_tick_gap * 0.4

    fig_param = Fig_Param(network_name, fig_name,
                          figsize, axis_title_font_size,
                          x_font_size,
                          y_font_size,
                          axis_tick_gap,
                          timestamp_split_cross_mark_offset)

    return fig_param


def main():
    """
    generate a matrix containing the existence of edges over time.
    plot it!
    """
    common_path = f'data/'

    network_name_list = [
        'US Legislative',                               # 0
        'Canadian Vote',                                # 1
        'UN Trade',                                     # 2
        'UN Vote',                                      # 3
        'Reddit',                                       # 4
        'Wikipedia',                                    # 5
        'Enron',                                        # 6
        'MOOC',                                         # 7
        'UCI',                                          # 8
        'Social Evo.',                                  # 9
        'Flights',                                      # 10
        'LastFM',                                       # 11
    ]
    
    edgelist_filename_list = [
        "LegisEdgelist.txt",                            # 0
        "canVote_edgelist.txt",                         # 1
        "UNtrade_edgelist.txt",                         # 2
        "UNvote_edgelist.txt",                          # 3
        "ml_reddit.csv",                                # 4
        "ml_wikipedia.csv",                             # 5
        "ml_enron.csv",                                 # 6
        "ml_mooc.csv",                                  # 7
        "ml_uci.csv",                                   # 8
        "ml_socialevolve.csv",                          # 9
        "covid_20200301_20200630_ext.csv",              # 10
        "ml_lastfm.csv",                                # 11
    ]
    directed = True  # whether to considered network as directed or not
    add_frame = True
    fig_index_list = [i for i in range(12)]  # i for i in range(12)
    for i in fig_index_list:
        edgelist_filename = f"{common_path}/{edgelist_filename_list[i]}"
        fig_name = f"figs/TET/{edgelist_filename_list[i].split('.')[0]}.png"  # png
        network_name = network_name_list[i]

        print("Info: Network name:", network_name)
        print("Info: Edge list file name:", edgelist_filename)
        from_temporal_edgelist_to_edge_presence_plot(edgelist_filename, network_name, fig_name,
                                                     add_frame=add_frame, directed=directed)


if __name__ == '__main__':
    main()
