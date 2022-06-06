"""
TEA Plots
"""

import sys

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    """
    print("Info: Interval size:", interval_size)

    u_idx = 1
    v_idx = 2
    ts_idx = 3

    temporal_edgelist = {}
    total_n_edges = 0
    with open(fname) as f:
        s = next(f)  # skip the first line
        for idx, line in enumerate(f):
            total_n_edges += 1
            e = line.strip().split(',')
            u = e[u_idx]  # source node
            v = e[v_idx]  # destination node
            ts = float(e[ts_idx])  # timestamp
            ts_bin_id = int(ts / interval_size)
            if ts_bin_id not in temporal_edgelist:
                temporal_edgelist[ts_bin_id] = {}
                temporal_edgelist[ts_bin_id][(u, v)] = 1
            else:
                if (u, v) not in temporal_edgelist[ts_bin_id]:
                    temporal_edgelist[ts_bin_id][(u, v)] = 1
                else:
                    temporal_edgelist[ts_bin_id][(u, v)] += 1

    print("Loading edge-list: Maximum timestamp is ", ts)
    print("Loading edge-list: Maximum timestamp-bin-id is", ts_bin_id)
    print("Loading edge-list: Total number of edges:", total_n_edges)
    return temporal_edgelist


def process_edgelist_per_timestamp(temp_edgelist):
    # generate distribution of the edges history
    unique_ts = list(temp_edgelist.keys())
    # unique_ts.sort()
    print(f"There are {len(unique_ts)} timestamps.")

    # get node set & total number of nodes
    node_dict = {}
    for t, e_dict in temp_edgelist.items():
        for e, exist in e_dict.items():
            if e[0] not in node_dict:
                node_dict[e[0]] = 1
            if e[1] not in node_dict:
                node_dict[e[1]] = 1
    num_nodes = len(node_dict)
    num_e_fully_connected = num_nodes * (num_nodes - 1)

    edge_frequency_dict = {}  # how many times an edge is seen
    ts_edges_dist = []  # contains different features specifying the characteristics of the edge distribution over time
    ts_edges_dist_density = []
    for curr_t in unique_ts:
        prev_ts = [ts for ts in unique_ts if ts < curr_t]
        edges_in_prev_ts = {}
        for bts in prev_ts:
            edges_in_prev_ts.update(temp_edgelist[bts])

        curr_ts_edge_list = temp_edgelist[curr_t]
        for e in curr_ts_edge_list:
            if e not in edge_frequency_dict:
                edge_frequency_dict[e] = 1
            else:
                edge_frequency_dict[e] += 1

        if len(curr_ts_edge_list) > 0:
            curr_ts_edges_dist = {'ts': curr_t,
                                  'new': len([e for e in curr_ts_edge_list if e not in edges_in_prev_ts]),
                                  'repeated': len([e for e in curr_ts_edge_list if e in edges_in_prev_ts]),
                                  'not_repeated': len([e for e in edges_in_prev_ts if e not in curr_ts_edge_list]),
                                  'total_curr_ts': len(curr_ts_edge_list),
                                  'total_seen_until_curr_ts': len(edges_in_prev_ts) + len(curr_ts_edge_list)
                                  }
            curr_ts_edges_dist_density = {'ts': curr_t,
                                          'new': (curr_ts_edges_dist['new'] * 1.0) / num_e_fully_connected,
                                          'repeated': (curr_ts_edges_dist['repeated'] * 1.0) / num_e_fully_connected,
                                          'not_repeated': (curr_ts_edges_dist[
                                                               'not_repeated'] * 1.0) / num_e_fully_connected,
                                          'total_curr_ts': (curr_ts_edges_dist[
                                                                'total_curr_ts'] * 1.0) / num_e_fully_connected,
                                          'total_seen_until_curr_ts': (curr_ts_edges_dist[
                                                                           'total_seen_until_curr_ts'] * 1.0) / num_e_fully_connected,
                                          }
        else:
            curr_ts_edges_dist = {'ts': curr_t,
                                  'new': 0,
                                  'repeated': 0,
                                  'not_repeated': 0,
                                  'total_curr_ts': 0,
                                  'total_seen_until_curr_ts': len(edges_in_prev_ts) + len(curr_ts_edge_list)
                                  }
            curr_ts_edges_dist_density = {'ts': curr_t,
                                          'new': 0,
                                          'repeated': 0,
                                          'not_repeated': 0,
                                          'total_curr_ts': 0,
                                          'total_seen_until_curr_ts': 0,
                                          }
        ts_edges_dist.append(curr_ts_edges_dist)
        ts_edges_dist_density.append(curr_ts_edges_dist_density)
    return ts_edges_dist, ts_edges_dist_density, edge_frequency_dict


def load_UN_temporarl_edgelist(fname):
    """
    treat each year as a timestamp
    """
    edgelist = open(fname, "r")
    edgelist.readline()
    lines = list(edgelist.readlines())
    edgelist.close()

    temp_edgelist = {}
    total_edges = 0

    for i in range(0, len(lines)):
        line = lines[i]
        values = line.split(',')
        t = int(values[0])
        u = values[1]
        v = values[2]
        if t not in temp_edgelist:
            temp_edgelist[t] = {}
            temp_edgelist[t][(u, v)] = 1
        else:
            if (u, v) not in temp_edgelist[t]:
                temp_edgelist[t][(u, v)] = 1
            else:
                temp_edgelist[t][(u, v)] += 1
        total_edges += 1
    print("Number of loaded edges: " + str(total_edges))
    print("Available timestamps: ", temp_edgelist.keys())
    return temp_edgelist


def load_flight_edgelist(fname):
    """
    treat each year as a timestamp
    """
    edgelist = open(fname, "r")
    edgelist.readline()
    lines = list(edgelist.readlines())
    edgelist.close()

    temp_edgelist = {}
    total_edges = 0

    for i in range(0, len(lines)):
        line = lines[i]
        values = line.split(',')
        t = values[-1].strip('\n')
        u = values[0]
        v = values[1]
        if t not in temp_edgelist:
            temp_edgelist[t] = {}
            temp_edgelist[t][(u, v)] = 1
        else:
            if (u, v) not in temp_edgelist[t]:
                temp_edgelist[t][(u, v)] = 1
            else:
                temp_edgelist[t][(u, v)] += 1
        total_edges += 1
    print("Number of loaded edges: " + str(total_edges))
    print("Available timestamps:", temp_edgelist.keys())
    return temp_edgelist


def load_retweet_edgelist(fname):
    """
    treat each year as a timestamp
    """
    edgelist = open(fname, "r")
    edgelist.readline()
    lines = list(edgelist.readlines())
    edgelist.close()

    temp_edgelist = {}
    total_edges = 0

    for i in range(0, len(lines)):
        line = lines[i]
        values = line.split(',')
        t = values[0]
        u = values[1]
        v = values[2].strip('\n')
        if t not in temp_edgelist:
            temp_edgelist[t] = {}
            temp_edgelist[t][(u, v)] = 1
        else:
            if (u, v) not in temp_edgelist[t]:
                temp_edgelist[t][(u, v)] = 1
            else:
                temp_edgelist[t][(u, v)] += 1
        total_edges += 1
    print("Number of loaded edges: " + str(total_edges))
    print("Available timestamps:", temp_edgelist.keys())
    return temp_edgelist


def plot_edges_bar_extended(ts_edges_dist, network_name, common_path, USLegis=False, plot_density='', extended_plot=False):
    ts_edges_dist_df = pd.DataFrame(ts_edges_dist, columns=['ts', 'new', 'repeated',
                                                            'not_repeated',
                                                            'total_curr_ts',
                                                            'total_seen_until_curr_ts'])
    # fig, ax = plt.subplots(figsize=(6, 5))
    fig, ax = plt.subplots(figsize=(10, 8))  # for covid
    plt.subplots_adjust(bottom=0.2, left=0.2)
    font_size = 20
    ticks_font_size = 19

    timestamps = ts_edges_dist_df['ts'].tolist()
    if (USLegis):
        timestamps = [int(j) + 97 for j in timestamps]
    if network_name == "Flight Mar to June 2020":
            timestamps = [j[5:] for j in timestamps]

    new = ts_edges_dist_df['new'].tolist()
    repeated = ts_edges_dist_df['repeated'].tolist()
    not_repeated = ts_edges_dist_df['not_repeated'].tolist()
    total_curr_ts = ts_edges_dist_df['total_curr_ts'].tolist()
    total_seen_so_far = ts_edges_dist_df['total_seen_until_curr_ts'].tolist()
    if not extended_plot:  # new & repeated
        plt.bar(timestamps, repeated, label='Repeated', color='#bababa', alpha=0.7)
        plt.bar(timestamps, new, label='New', bottom=repeated, color='#ca0020', alpha=0.9, hatch='//')

        plt.axvline(x=(timestamps[int(0.85 * len(timestamps))]), color="blue", linestyle="--", linewidth=2)
        plt.text((timestamps[int(0.85 * len(timestamps))]), 0,
                 'x', va='center', ha='center', fontsize=font_size, fontweight='heavy', color='blue')
    else:  # add not_repeated as well
        _x = np.arange(len(timestamps))  # numerical x axis
        rects = plt.bar(_x - 0.2, total_curr_ts, width=0.4, color="#3288bd")
        plt.bar(_x + 0.2, not_repeated, width=0.4, label='not repeated', color="#da3287")  # 5dbce4
        plt.bar(_x - 0.2, repeated, width=0.4, label='Repeated', color="#99d594")
        plt.bar(_x - 0.2, new, width=0.4, label='New', color="#fc8d59", bottom=repeated)
        plt.xticks(_x, timestamps, fontsize=ticks_font_size)
        plt.axvline(x=(_x[int(0.85 * len(_x))]), color="black")
        plt.text((_x[int(0.75 * len(_x))]), rects[int(0.15 * len(_x))].get_height() + 10,
                 'test split', ha='center')

    # if the axis label is a string
    if isinstance(timestamps[0], str):
        # get labels once in 15
        labels = []
        time_gap = 15
        for i in range(len(timestamps)):
            if i % time_gap == 0:
                labels.append(timestamps[i])

        time_gaps = list(range(0, len(timestamps), time_gap))
        plt.xticks(time_gaps, labels, fontsize=ticks_font_size)
        plt.yticks(fontsize=ticks_font_size)

    plt.margins(x=0)
    # plt.title(network_name)
    plt.xlabel("Timestamp", fontsize=font_size)
    plt.ylabel("Number of edges", fontsize=font_size)
    plt.legend()
    plt.savefig(f"{common_path}/figs/dist/{network_name}{plot_density}" + ".pdf")
    plt.close()


def plot_edges_bar(ts_edges_dist, network_name, common_path, USLegis=False):
    ts_edges_dist_df = pd.DataFrame(ts_edges_dist, columns=['ts', 'new', 'repeated',
                                                            'not_repeated',
                                                            'total_curr_ts',
                                                            'total_seen_until_curr_ts'])
    ### Additional Stats ###
    mean = ts_edges_dist_df.mean(axis=0)
    print("INFO: Network Name:", network_name)
    print("INFO: AVG. stats. over all timestamps: ", mean)
    print("INFO: ratio of avg.(new)/avg.(total_curr_ts): {:.2f}".format(mean['new'] / mean['total_curr_ts']))
    ###

    fig, ax = plt.subplots(figsize=(7, 5))  # lastfm, mooc, reddit, UNtrade, UNvote
    fig, ax = plt.subplots(figsize=(7, 5))  # covid
    # fig, ax = plt.subplots(figsize=(6, 4.5))  # others
    plt.subplots_adjust(bottom=0.2, left=0.2)
    font_size = 20 # 18
    ticks_font_size = 18 # 18

    timestamps = ts_edges_dist_df['ts'].tolist()
    if (USLegis):
        timestamps = [int(j) + 97 for j in timestamps]
    if network_name == "Flight Mar to June 2020":
            timestamps = [j[5:] for j in timestamps]

    new = ts_edges_dist_df['new'].tolist()
    repeated = ts_edges_dist_df['repeated'].tolist()

    # plotting stuffs
    # bar plot
    plt.bar(timestamps, repeated, label='Repeated', color='#404040', alpha=0.4)
    plt.bar(timestamps, new, label='New', bottom=repeated, color='#ca0020', alpha=0.8, hatch='//')
    # test split line
    plt.axvline(x=(timestamps[int(0.85 * len(timestamps))]), color="blue", linestyle="--", linewidth=2)
    plt.text((timestamps[int(0.85 * len(timestamps))]), 0,
             'x', va='center', ha='center', fontsize=font_size, fontweight='heavy', color='blue')

    # axis processing
    # if the axis label is a string
    if isinstance(timestamps[0], str):
        # get labels once in 15
        labels = []
        time_gap = 15
        for i in range(len(timestamps)):
            if i % time_gap == 0:
                labels.append(timestamps[i])

        time_gaps = list(range(0, len(timestamps), time_gap))
        plt.xticks(time_gaps, labels, fontsize=ticks_font_size)
        plt.yticks(fontsize=ticks_font_size)

    plt.margins(x=0)
    plt.xlabel("Timestamp", fontsize=font_size)
    plt.ylabel("Number of edges", fontsize=font_size)
    plt.legend()
    plt.savefig(f"figs/TEA/{network_name}" + ".pdf")
    plt.close()


def plot_UNvote(args):
    # dataset with discrete timestamp
    edgelist_filename = f'{args.common_path}/UNvote_edgelist.txt'
    network_name = "UN Vote"
    temp_edgelist = load_UN_temporarl_edgelist(edgelist_filename)
    gen_plot_edge_dist(temp_edgelist, network_name, args, USLegis=False)


def plot_UNtrade(args):
    # dataset with discrete timestamp
    edgelist_filename = f'{args.common_path}/UNtrade_edgelist.txt'
    network_name = "UN Trade"
    temp_edgelist = load_UN_temporarl_edgelist(edgelist_filename)
    gen_plot_edge_dist(temp_edgelist, network_name, args, USLegis=False)


def plot_Canvote(args):
    edgelist_filename = f'{args.common_path}/canVote_edgelist.txt'
    network_name = "Canadian Vote"
    temp_edgelist = load_UN_temporarl_edgelist(edgelist_filename)
    gen_plot_edge_dist(temp_edgelist, network_name, args, USLegis=False)


def plot_USLegis(args):
    edgelist_filename = f'{args.common_path}/LegisEdgelist.txt'
    network_name = "US Legislative"
    temp_edgelist = load_UN_temporarl_edgelist(edgelist_filename)
    gen_plot_edge_dist(temp_edgelist, network_name, args, USLegis=True)


def plot_continuous_network(args, network_name):
    if network_name in ['lastfm', 'enron']:
        interval_size = 86400 * 30  # one month
    elif network_name in [ 'uci', 'socialevolve']:
        interval_size = 86400 * 5  # 5 days
    else:
        interval_size = 86400  # one day
    edgelist_filename = f'{args.common_path}/ml_{network_name}.csv'
    temp_edgelist = load_continuous_edgelist(edgelist_filename, interval_size=interval_size)
    gen_plot_edge_dist(temp_edgelist, network_name, args, USLegis=False)

def plot_flight(args):
    edgelist_filename = f"{args.common_path}/covid_20200301_20200630_ext.csv"
    network_name = "Flight Mar to June 2020"
    temp_edgelist = load_flight_edgelist(edgelist_filename)
    gen_plot_edge_dist(temp_edgelist, network_name, args, USLegis=False)


def plot_retweet(args):
    edgelist_filename = f"{args.common_path}/tg_polisci_retweet.csv"
    network_name = "Political Science Retweet Network"
    temp_edgelist = load_retweet_edgelist(edgelist_filename)
    gen_plot_edge_dist(temp_edgelist, network_name, args, USLegis=False)


def gen_plot_edge_dist(temp_edgelist, network_name, args, USLegis):
    ts_edges_dist, ts_edges_dist_density, edge_frequency_dict = process_edgelist_per_timestamp(temp_edgelist)
    plot_edges_bar(ts_edges_dist, network_name, args.common_path, USLegis=USLegis)



def main():
    """
    Test is last 15% of timestamps
    draw a line there
    """

    class my_args():
        def __init__(self, network_name, ext_plt):
            self.network_name = network_name
            self.ext_plt = ext_plt
            self.common_path = f'data/'

    network_list = [
        'USLegis',
        'wikipedia',
        'covid',
        'canVote',
        'reddit',
        'UNVote',
        'UNtrade',
        'mooc',
        'uci',
        'enron',
        'socialevolve',
        'lastfm'
    ]
    ext_plt = False
    for net_name in network_list:
        args = my_args(net_name, ext_plt)
        print("Args:", vars(args))
        if net_name == 'covid':
            plot_flight(args)
        elif net_name == 'retweet':
            plot_retweet(args)
        elif net_name == 'USLegis':
            plot_USLegis(args)
        elif net_name == 'UNVote':
            plot_UNvote(args)
        elif net_name == 'UNtrade':
            plot_UNtrade(args)
        elif net_name == 'canVote':
            plot_Canvote(args)
        else:
            plot_continuous_network(args, net_name)


if __name__ == "__main__":
    main()
