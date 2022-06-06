from numba import jit
import numpy as np
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


@jit(nopython=True)
def nodets2key(batch: int, node: int, ts: float):
    key = '-'.join([str(batch), str(node), float2str(ts)])
    return key


@jit(nopython=True)
def float2str(ts):
    return str(int(round(ts)))


def make_batched_keys(node_record, t_record):
    batch = node_record.shape[0]
    support = node_record.shape[1]
    batched_keys = make_batched_keys_l(node_record, t_record, batch, support)
    batched_keys = np.array(batched_keys).reshape((batch, support))
    # batched_keys = np.array([nodets2key(b, n, t) for b, n, t in zip(batch_matrix.ravel(), node_record.ravel(), t_record.ravel())]).reshape(batch, support)
    return batched_keys


@jit(nopython=True)
def make_batched_keys_l(node_record, t_record, batch, support):
    batch_matrix = np.arange(batch).repeat(support).reshape((-1, support))
    # batch_matrix = np.tile(np.expand_dims(np.arange(batch), 1), (1, support))
    batched_keys = []
    for i in range(batch):
        for j in range(support):
            b = batch_matrix[i, j]
            n = node_record[i, j]
            t = t_record[i, j]
            batched_keys.append(nodets2key(b, n, t))
    return batched_keys