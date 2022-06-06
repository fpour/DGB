import math
import random
import torch
import pandas as pd
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from sklearn.metrics import *


### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1

        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.neg_sample = 'rnd'  # negative edge sampling method: random
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class RandEdgeSampler_NRE(object):
    """
    ~ "history"
    Random Negative Edge Sampling: NRE: "Non-Repeating Edges" are randomly sampled to make task more complicated
    Note: the edge history is constructed in a way that it inherently preserve the direction information
    Note: we consider that randomly sampled edges come from two sources:
      1. some are randomly sampled from all possible pairs of edges
      2. some are randomly sampled from edges seen before but are not repeating in current batch
    """

    def __init__(self, src_list, dst_list, ts_list, seed=None, rnd_sample_ratio=0):
        """
        'src_list', 'dst_list', 'ts_list' are related to the full data! All possible edges in train, validation, test
        """
        self.seed = None
        self.neg_sample = 'nre'  # negative edge sampling method: non-repeating edges
        self.rnd_sample_ratio = rnd_sample_ratio
        self.src_list = src_list
        self.dst_list = dst_list
        self.ts_list = ts_list
        self.src_list_distinct = np.unique(src_list)
        self.dst_list_distinct = np.unique(dst_list)
        self.ts_list_distinct = np.unique(ts_list)
        self.ts_init = min(self.ts_list_distinct)

        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
            self.random_state = np.random.RandomState(self.seed)

    def get_edges_in_time_interval(self, start_ts, end_ts):
        """
        return edges of a specific time interval
        """
        valid_ts_interval = (self.ts_list >= start_ts) * (self.ts_list <= end_ts)
        interval_src_l = self.src_list[valid_ts_interval]
        interval_dst_l = self.dst_list[valid_ts_interval]
        interval_edges = {}
        for src, dst in zip(interval_src_l, interval_dst_l):
            if (src, dst) not in interval_edges:
                interval_edges[(src, dst)] = 1
        return interval_edges

    def get_difference_edge_list(self, first_e_set, second_e_set):
        """
        return edges in the first_e_set that are not in the second_e_set
        """
        difference_e_set = set(first_e_set) - set(second_e_set)
        src_l, dst_l = [], []
        for e in difference_e_set:
            src_l.append(e[0])
            dst_l.append(e[1])
        return np.array(src_l), np.array(dst_l)

    def sample(self, size, current_split_start_ts, current_split_end_ts):
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
        non_repeating_e_src_l, non_repeating_e_dst_l = self.get_difference_edge_list(history_e_dict,
                                                                                     current_split_e_dict)

        num_smp_rnd = int(self.rnd_sample_ratio * size)
        num_smp_from_hist = size - num_smp_rnd
        if num_smp_from_hist > len(non_repeating_e_src_l):
            num_smp_from_hist = len(non_repeating_e_src_l)
            num_smp_rnd = size - num_smp_from_hist

        replace = len(self.src_list_distinct) < num_smp_rnd
        rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)

        replace = len(self.dst_list_distinct) < num_smp_rnd
        rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)

        replace = len(non_repeating_e_src_l) < num_smp_from_hist
        nre_e_index = np.random.choice(len(non_repeating_e_src_l), size=num_smp_from_hist, replace=replace)

        negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], non_repeating_e_src_l[nre_e_index]])
        negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], non_repeating_e_dst_l[nre_e_index]])

        return negative_src_l, negative_dst_l

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class RandEdgeSampler_adversarial(object):
  """
  Adversarial Random Edge Sampling as Negative Edges
  RandEdgeSampler_adversarial(src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE, seed=None, rnd_sample_ratio=0)
  """

  def __init__(self, src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE, seed=None, rnd_sample_ratio=0):
    """
    'src_list', 'dst_list', 'ts_list' are related to the full data! All possible edges in train, validation, test
    """
    if not (NEG_SAMPLE == 'hist_nre' or NEG_SAMPLE == 'induc_nre'):
      raise ValueError("Undefined Negative Edge Sampling Strategy!")

    self.seed = None
    self.neg_sample = NEG_SAMPLE
    self.rnd_sample_ratio = rnd_sample_ratio
    self.src_list = src_list
    self.dst_list = dst_list
    self.ts_list = ts_list
    self.src_list_distinct = np.unique(src_list)
    self.dst_list_distinct = np.unique(dst_list)
    self.ts_list_distinct = np.unique(ts_list)
    self.ts_init = min(self.ts_list_distinct)
    self.ts_end = max(self.ts_list_distinct)
    self.ts_test_split = last_ts_train_val
    self.e_train_val_l = self.get_edges_in_time_interval(self.ts_init, self.ts_test_split)

    if seed is not None:
      self.seed = seed
      np.random.seed(self.seed)
      self.random_state = np.random.RandomState(self.seed)

  def get_edges_in_time_interval(self, start_ts, end_ts):
    """
    return edges of a specific time interval
    """
    valid_ts_interval = (self.ts_list >= start_ts) * (self.ts_list <= end_ts)
    interval_src_l = self.src_list[valid_ts_interval]
    interval_dst_l = self.dst_list[valid_ts_interval]
    interval_edges = {}
    for src, dst in zip(interval_src_l, interval_dst_l):
      if (src, dst) not in interval_edges:
        interval_edges[(src, dst)] = 1
    return interval_edges

  def get_difference_edge_list(self, first_e_set, second_e_set):
    """
    return edges in the first_e_set that are not in the second_e_set
    """
    difference_e_set = set(first_e_set) - set(second_e_set)
    src_l, dst_l = [], []
    for e in difference_e_set:
      src_l.append(e[0])
      dst_l.append(e[1])
    return np.array(src_l), np.array(dst_l)

  def sample(self, size, current_split_start_ts, current_split_end_ts):
    if self.neg_sample == 'hist_nre':
      negative_src_l, negative_dst_l = self.sample_hist_NRE(size, current_split_start_ts, current_split_end_ts)
    elif self.neg_sample == 'induc_nre':
      negative_src_l, negative_dst_l = self.sample_induc_NRE(size, current_split_start_ts, current_split_end_ts)
    else:
      raise ValueError("Undefined Negative Edge Sampling Strategy!")
    return negative_src_l, negative_dst_l

  def sample_hist_NRE(self, size, current_split_start_ts, current_split_end_ts):
    """
    method one:
    "historical adversarial sampling": (~ inductive historical edges)
    randomly samples among previously seen edges that are not repeating in current batch,
    fill in any remaining with randomly sampled
    """
    history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
    current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
    non_repeating_e_src_l, non_repeating_e_dst_l = self.get_difference_edge_list(history_e_dict,
                                                                                 current_split_e_dict)
    num_smp_rnd = int(self.rnd_sample_ratio * size)
    num_smp_from_hist = size - num_smp_rnd
    if num_smp_from_hist > len(non_repeating_e_src_l):
      num_smp_from_hist = len(non_repeating_e_src_l)
      num_smp_rnd = size - num_smp_from_hist

    replace = len(self.src_list_distinct) < num_smp_rnd
    rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)

    replace = len(self.dst_list_distinct) < num_smp_rnd
    rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)

    replace = len(non_repeating_e_src_l) < num_smp_from_hist
    nre_e_index = np.random.choice(len(non_repeating_e_src_l), size=num_smp_from_hist, replace=replace)

    negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], non_repeating_e_src_l[nre_e_index]])
    negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], non_repeating_e_dst_l[nre_e_index]])

    return negative_src_l, negative_dst_l

  def sample_induc_NRE(self, size, current_split_start_ts, current_split_end_ts):
    """
    method two:
    "inductive adversarial sampling": (~ inductive non repeating edges)
    considers only edges that have been seen (in red region),
    fill in any remaining with randomly sampled
    """
    history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
    current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
    induc_adversarial_e = set(set(history_e_dict) - set(self.e_train_val_l)) - set(current_split_e_dict)
    induc_adv_src_l, induc_adv_dst_l = [], []
    if len(induc_adversarial_e) > 0:
      for e in induc_adversarial_e:
        induc_adv_src_l.append(int(e[0]))
        induc_adv_dst_l.append(int(e[1]))
      induc_adv_src_l = np.array(induc_adv_src_l)
      induc_adv_dst_l = np.array(induc_adv_dst_l)

    num_smp_rnd = size - len(induc_adversarial_e)

    if num_smp_rnd > 0:
      replace = len(self.src_list_distinct) < num_smp_rnd
      rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)
      replace = len(self.dst_list_distinct) < num_smp_rnd
      rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)

      negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], induc_adv_src_l])
      negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], induc_adv_dst_l])
    else:
      rnd_induc_hist_index = np.random.choice(len(induc_adversarial_e), size=size, replace=False)
      negative_src_l = induc_adv_src_l[rnd_induc_hist_index]
      negative_dst_l = induc_adv_dst_l[rnd_induc_hist_index]

    return negative_src_l, negative_dst_l

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def eval_one_epoch_original(hint, tgan, sampler, src, dst, ts, label, NUM_NEIGHBORS):
    val_ap, val_auc_roc = [], []
    measures_list = []
    with torch.no_grad():
        tgan = tgan.eval()
        # TEST_BATCH_SIZE = 30
        TEST_BATCH_SIZE = 200
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast_original(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc_roc.append(roc_auc_score(true_label, pred_score))

            # extra performance measures
            measures_dict = extra_measures(true_label, pred_score)
            measures_list.append(measures_dict)
        measures_df = pd.DataFrame(measures_list)
        avg_measures_dict = measures_df.mean()

    return np.mean(val_ap), np.mean(val_auc_roc), avg_measures_dict


def eval_one_epoch_modified(hint, tgan, sampler, src, dst, ts, label, NUM_NEIGHBORS):
    val_ap, val_auc_roc = [], []
    measures_list = []
    with torch.no_grad():
        tgan = tgan.eval()
        # TEST_BATCH_SIZE = 30
        TEST_BATCH_SIZE = 200
        num_test_instance = len(src)
        num_test_batch = int(math.ceil(num_test_instance / TEST_BATCH_SIZE))
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)

            if sampler.neg_sample != 'rnd':
                negative_samples_sources, negative_samples_destinations = \
                    sampler.sample(size, ts_l_cut[0], ts_l_cut[-1])
            else:
                negative_samples_sources, negative_samples_destinations = sampler.sample(size)
                negative_samples_sources = src_l_cut

            # contrast_modified(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20)
            pos_prob = tgan.contrast_modified(src_l_cut, dst_l_cut, ts_l_cut, NUM_NEIGHBORS)
            neg_prob = tgan.contrast_modified(negative_samples_sources, negative_samples_destinations, ts_l_cut, NUM_NEIGHBORS)

            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc_roc.append(roc_auc_score(true_label, pred_score))

            # extra performance measures
            measures_dict = extra_measures(true_label, pred_score)
            measures_list.append(measures_dict)
        measures_df = pd.DataFrame(measures_list)
        avg_measures_dict = measures_df.mean()

    return np.mean(val_ap), np.mean(val_auc_roc), avg_measures_dict


def get_measures_for_threshold(y_true, y_pred_score, threshold):
    """
    compute measures for a specific threshold
    """
    perf_measures = {}
    y_pred_label = y_pred_score > threshold
    perf_measures['acc'] = accuracy_score(y_true, y_pred_label)
    prec, rec, f1, num = precision_recall_fscore_support(y_true, y_pred_label, average='binary', zero_division=1)
    perf_measures['prec'] = prec
    perf_measures['rec'] = rec
    perf_measures['f1'] = f1
    return perf_measures


def extra_measures(y_true, y_pred_score):
    """
    compute extra performance measures
    """
    perf_dict = {}
    # find optimal threshold of au-roc
    perf_dict['ap'] = average_precision_score(y_true, y_pred_score)

    perf_dict['au_roc_score'] = roc_auc_score(y_true, y_pred_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_score)
    opt_idx = np.argmax(tpr - fpr)
    opt_thr_auroc = roc_thresholds[opt_idx]
    perf_dict['opt_thr_au_roc'] = opt_thr_auroc
    auroc_perf_dict = get_measures_for_threshold(y_true, y_pred_score, opt_thr_auroc)
    perf_dict['acc_auroc_opt_thr'] = auroc_perf_dict['acc']
    perf_dict['prec_auroc_opt_thr'] = auroc_perf_dict['prec']
    perf_dict['rec_auroc_opt_thr'] = auroc_perf_dict['rec']
    perf_dict['f1_auroc_opt_thr'] = auroc_perf_dict['f1']

    prec_pr_curve, rec_pr_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_score)
    perf_dict['au_pr_score'] = auc(rec_pr_curve, prec_pr_curve)
    # convert to f score
    fscore = (2 * prec_pr_curve * rec_pr_curve) / (prec_pr_curve + rec_pr_curve)
    opt_idx = np.argmax(fscore)
    opt_thr_aupr = pr_thresholds[opt_idx]
    perf_dict['opt_thr_au_pr'] = opt_thr_aupr
    aupr_perf_dict = get_measures_for_threshold(y_true, y_pred_score, opt_thr_aupr)
    perf_dict['acc_aupr_opt_thr'] = aupr_perf_dict['acc']
    perf_dict['prec_aupr_opt_thr'] = aupr_perf_dict['prec']
    perf_dict['rec_aupr_opt_thr'] = aupr_perf_dict['rec']
    perf_dict['f1_aupr_opt_thr'] = aupr_perf_dict['f1']

    # threshold = 0.5
    perf_half_dict = get_measures_for_threshold(y_true, y_pred_score, 0.5)
    perf_dict['acc_thr_0.5'] = perf_half_dict['acc']
    perf_dict['prec_thr_0.5'] = perf_half_dict['prec']
    perf_dict['rec_thr_0.5'] = perf_half_dict['rec']
    perf_dict['f1_thr_0.5'] = perf_half_dict['f1']

    return perf_dict
