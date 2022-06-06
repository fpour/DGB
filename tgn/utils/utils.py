import numpy as np
import torch


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
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

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.neg_sample = 'rnd'  # negative edge sampling method: random edges
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
        induc_adv_src_l.append(e[0])
        induc_adv_dst_l.append(e[1])
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


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    src_idx = src_idx.astype(np.int32)
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times