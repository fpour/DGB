"""
load a trained model for link prediction and test its performance on a set of positive and negative edges
"""

import pandas as pd
from log import *
from parser import *
from eval import *
from utils import *
from train import *
from module import CAWN
from graph import NeighborFinder
import resource
import time

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
POS_ENC = args.pos_enc
POS_DIM = args.pos_dim
WALK_POOL = args.walk_pool
WALK_N_HEAD = args.walk_n_head
WALK_MUTUAL = args.walk_mutual if WALK_POOL == 'attn' else False
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
AGG = args.agg
N_RUNS = args.n_runs
VAL_RATIO = args.val_ratio
TEST_RATIO = args.test_ratio
SEED = args.seed
NEG_SAMPLE = args.neg_sample
assert (CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

# Load data and sanity check
g_df = pd.read_csv('./data/ml_{}.csv'.format(DATA))
e_feat = np.load('./data/ml_{}.npy'.format(DATA))
n_feat = np.load('./data/ml_{}_node.npy'.format(DATA))
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
max_idx = max(src_l.max(), dst_l.max())
assert (np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx)  # all nodes except node 0 should appear and be
# compactly indexed
assert (n_feat.shape[0] == max_idx + 1)  # the nodes need to map one-to-one to the node feat matrix

# to refine node and edge features dimensions
# additional for CAW data specifically
if DATA in ['enron', 'socialevolve', 'uci', 'lastfm']:
    node_zero_padding = np.zeros((n_feat.shape[0], 172 - n_feat.shape[1]))
    n_feat = np.concatenate([n_feat, node_zero_padding], axis=1)
    edge_zero_padding = np.zeros((e_feat.shape[0], 172 - e_feat.shape[1]))
    e_feat = np.concatenate([e_feat, edge_zero_padding], axis=1)


# split and pack the data by generating valid train/val/test mask according to the "mode"
val_time, test_time = list(np.quantile(g_df.ts, [(1 - VAL_RATIO - TEST_RATIO), (1 - TEST_RATIO)]))
if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_flag = (ts_l <= val_time)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time

else:
    random.seed(2020)  # set the seed
    assert (args.mode == 'i')
    logger.info('Inductive training...')
    # pick some nodes to mask (i.e. reserved for testing) for inductive setting
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])),
                                      int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (
            none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
    valid_test_flag = (ts_l > test_time) * (
            none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
    valid_test_new_new_flag = (ts_l > test_time) * mask_src_flag * mask_dst_flag
    valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)
    logger.info(
        'Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))

# split data according to the mask
train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = src_l[valid_train_flag], \
                                                                     dst_l[valid_train_flag], ts_l[valid_train_flag], \
                                                                     e_idx_l[valid_train_flag], \
                                                                     label_l[valid_train_flag]
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = src_l[valid_val_flag], dst_l[valid_val_flag], \
                                                           ts_l[valid_val_flag], e_idx_l[valid_val_flag], \
                                                           label_l[valid_val_flag]
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], \
                                                                dst_l[valid_test_flag], ts_l[valid_test_flag], \
                                                                e_idx_l[valid_test_flag], label_l[valid_test_flag]
if args.mode == 'i':
    test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = \
        src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], \
        e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]
    test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = \
        src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], \
        e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]
train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)

# create two neighbor finders to handle graph extraction.
# for transductive mode all phases use full_ngh_finder, for inductive node train/val phases use the partial one
# while test phase still always uses the full one
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
partial_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
partial_ngh_finder = NeighborFinder(partial_adj_list, bias=args.bias, use_cache=NGH_CACHE,
                                    sample_method=args.pos_sample)
ngh_finders = partial_ngh_finder, full_ngh_finder

# create random samplers to generate train/val/test instances
# Set seeds for validation and testing so negatives are the same across different runs
# train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l,))
# val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l), seed=0)
# test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l), seed=1)
# rand_samplers = train_rand_sampler, val_rand_sampler

if NEG_SAMPLE != 'rnd':
    # RandEdgeSampler_adversarial(src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE, seed=None, rnd_sample_ratio=0)
    # train_rand_sampler = RandEdgeSampler_adversarial(train_src_l, train_dst_l, train_ts_l)
    # val_rand_sampler = RandEdgeSampler_adversarial(np.concatenate((train_src_l, val_src_l)),
    #                                        np.concatenate((train_dst_l, val_dst_l)),
    #                                        np.concatenate((train_ts_l, val_ts_l)), seed=0)
    test_rand_sampler = RandEdgeSampler_adversarial(np.concatenate((train_src_l, val_src_l, test_src_l)),
                                                    np.concatenate((train_dst_l, val_dst_l, test_dst_l)),
                                                    np.concatenate((train_ts_l, val_ts_l, test_ts_l)), val_ts_l[-1],
                                                    NEG_SAMPLE, seed=1)
    # rand_samplers = train_rand_sampler, val_rand_sampler
else:
    # train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l, ))
    # val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l), seed=0)
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l),
                                        (train_dst_l, val_dst_l, test_dst_l), seed=1)
    # rand_samplers = train_rand_sampler, val_rand_sampler


# multiprocessing memory setting
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (200 * args.bs, rlimit[1]))

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

logger.info("************************************")
logger.info("*********** Test starts *************")
start_test = time.time()

for i in range(N_RUNS):
    start_time_run = time.time()
    logger.info("************************************")
    logger.info("********** Run {} starts. **********".format(i))

    runtime_id = '{}-{}-{}'.format('CAWN', args.mode[0], args.agg)
    MODEL_SAVE_PATH = f'./saved_models/{runtime_id}-{args.data}-{i}.pth'

    # model initialization
    cawn = CAWN(n_feat, e_feat, agg=AGG,
                num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
                n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC,
                num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL,
                walk_linear_out=args.walk_linear_out,
                cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path)

    # load saved parameters of the model
    cawn.load_state_dict(torch.load(MODEL_SAVE_PATH))
    cawn = cawn.to(device)
    cawn.eval()

    # final testing
    cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
    test_acc, test_ap, test_f1, test_auc, test_avg_measures_dict = eval_one_epoch_modified(
        'test for {} nodes'.format(args.mode),
        cawn,
        test_rand_sampler, test_src_l,
        test_dst_l, test_ts_l,
        test_label_l, test_e_idx_l)
    logger.info('Test statistics: {} all nodes -- acc_inherent: {}'.format(args.mode, test_acc))
    logger.info('Test statistics: {} all nodes -- auc_inherent: {}'.format(args.mode, test_auc))
    logger.info('Test statistics: {} all nodes -- ap_inherent: {}'.format(args.mode, test_ap))
    for measure_name, measure_value in test_avg_measures_dict.items():
      logger.info('Test statistics: {} all nodes -- {}: {}'.format(args.mode, measure_name, measure_value))


    test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1] * 6
    if args.mode == 'i':
        test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc, test_new_new_avg_measures_dict = \
            eval_one_epoch_modified('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_new_l,
                                    test_dst_new_new_l, test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l)

        logger.info('Test statistics: {} new-new nodes -- acc_inherent: {}'.format(args.mode, test_new_new_acc))
        logger.info('Test statistics: {} new-new nodes -- auc_inherent: {}'.format(args.mode, test_new_new_auc))
        logger.info('Test statistics: {} new-new nodes -- ap_inherent: {}'.format(args.mode, test_new_new_ap))
        for measure_name, measure_value in test_new_new_avg_measures_dict.items():
            logger.info('Test statistics: {} new-new nodes -- {}: {}'.format(args.mode, measure_name, measure_value))


        test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc, test_new_old_avg_measures_dict = \
            eval_one_epoch_modified('test for {} nodes'.format(args.mode), cawn, test_rand_sampler,
                                    test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l,
                                    test_label_new_old_l, test_e_idx_new_old_l)
        logger.info('Test statistics: {} new-old nodes -- acc_inherent: {}'.format(args.mode, test_new_old_acc))
        logger.info('Test statistics: {} new-old nodes -- auc_inherent: {}'.format(args.mode, test_new_old_auc))
        logger.info('Test statistics: {} new-old nodes -- ap_inherent: {}'.format(args.mode, test_new_old_ap))
        for measure_name, measure_value in test_new_old_avg_measures_dict.items():
            logger.info('Test statistics: {} new-old nodes -- {}: {}'.format(args.mode, measure_name, measure_value))

    logger.info("Run {}, elapsed time: {} seconds.".format(i, str(time.time() - start_time_run)))
logger.info('Info: Total elapsed time: {} seconds.'.format(time.time() - start_test))
