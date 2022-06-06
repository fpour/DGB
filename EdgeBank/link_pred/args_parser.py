"""
Parsing arguments of different scripts
"""
import argparse
import sys


def parse_args_edge_bank():
    # argument passing
    parser = argparse.ArgumentParser(description='EdgeBank: A baseline link prediction approach for temporal graphs.')
    # parameters for data costruction (for the get_data(.) function)
    parser.add_argument('--val_ratio', type=float, default=0.15, help='validation ratio.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='test ratio.')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='whether using different new nodes for validation and test.')
    parser.add_argument('--randomize_features', action='store_true', help='whether using randomized initial features.')
    # parameters for specifying data & model
    parser.add_argument('-d', '--data', type=str, default='wiki', help='name of the network dataset.')
    parser.add_argument('--n_runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--mem_mode', type=str, default='unlim_mem', help='How memory of EdgeBank works.',
                        choices=['unlim_mem', 'repeat_freq', 'time_window'])
    parser.add_argument('--w_mode', type=str, default='fixed',
                        help='In time interval-based memory, how to select time window size.',
                        choices=['fixed', 'avg_reoccur'])
    parser.add_argument('--neg_sample', type=str, default='rnd', choices=['rnd', 'hist_nre', 'induc_nre'],
                        help='Strategy for the negative edge sampling.')

    try:
        args = parser.parse_args()
        print("Info: Arguments:\n", args)
    except:
        parser.print_help()
        sys.exit()

    return args
