import random
import numpy as np
from numba import jit
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


@jit(nopython=True)
def seq_binary_sample(ngh_binomial_prob, num_neighbor):
    sampled_idx = []
    for j in range(num_neighbor):
        idx = seq_binary_sample_one(ngh_binomial_prob)
        sampled_idx.append(idx)
    sampled_idx = np.array(sampled_idx)  # not necessary but just for type alignment with the other branch
    return sampled_idx


@jit(nopython=True)
def seq_binary_sample_one(ngh_binomial_prob):
    seg_len = 10
    a_l_seg = np.random.random((seg_len,))
    seg_idx = 0
    for idx in range(len(ngh_binomial_prob)-1, -1, -1):
        a = a_l_seg[seg_idx]
        seg_idx += 1 # move one step forward
        if seg_idx >= seg_len:
            a_l_seg = np.random.random((seg_len,))  # regenerate a batch of new random values
            seg_idx = 0  # and reset the seg_idx
        if a < ngh_binomial_prob[idx]:
            # print('=' * 50)
            # print(a, len(ngh_binomial_prob) - idx, len(ngh_binomial_prob),
            #       (len(ngh_binomial_prob) - idx) / len(ngh_binomial_prob), ngh_binomial_prob)
            return idx
    return 0  # very extreme case due to float rounding error


@jit(nopython=True)
def bisect_left_adapt(a, x):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    lo = 0
    hi = len(a)
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < x: lo = mid+1
        else: hi = mid
    return lo