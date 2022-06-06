"""
preparing our own data to be used by different models
"""
import pandas as pd
from datetime import datetime as dt
import time
import numpy as np
from pathlib import Path


def prepare_dataset(edgelist_filename, network_name):
    """
    prepare dataset
    dataset is one of the following: canVote, LegisEdgelist, UNtrade, UNvote
    """
    edgelist_df = pd.read_csv(edgelist_filename)
    edgelist_df.columns = ['t', 'u', 'v', 'w']

    # processing timestamps
    if network_name != 'LegisEdgelist':
        year_l = edgelist_df['t'].tolist()  # t is in 'year'
        dates_list = [time.mktime(dt.strptime(str(date_str), '%Y').timetuple()) for date_str in year_l]  # linux format
        min_date = np.min(dates_list)
        cardinal_date_l = [(ts - min_date) for ts in dates_list]
    else:
        cardinal_date_l = edgelist_df['t'].tolist()

    # processing node ids
    node_list = list(set(edgelist_df['u'].tolist() + edgelist_df['v'].tolist()))
    u_id_l = [node_list.index(u) for u in edgelist_df['u'].tolist()]
    v_id_l = [node_list.index(v) for v in edgelist_df['v'].tolist()]

    # dummy state label
    state_label_l = [0 for _ in range(len(u_id_l))]

    processed_df = pd.DataFrame(zip(u_id_l, v_id_l, cardinal_date_l, state_label_l, edgelist_df['w'].tolist()),
                                columns=['u_id', 'v_id', 'timestamp', 'state_label', 'w'])
    saved_filename = edgelist_filename.split('.txt')[0] + '.csv'
    processed_df.to_csv(saved_filename, index=False)


def main():
    """
    commands to run
    """
    network_name = 'UNtrade'
    common_path = Path(__file__).parents[1]
    print("Info: common_path:", common_path)
    if network_name != 'LegisEdgelist':
        edgelist_filename = f'{common_path}/ebank/data/{network_name}_edgelist.txt'
    else:
        edgelist_filename = f'{common_path}/ebank/data/{network_name}.txt'

    prepare_dataset(edgelist_filename, network_name)


if __name__ == '__main__':
    main()