import math
import torch
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.metrics import average_precision_score
from sklearn.metrics import *
from sklearn.metrics import roc_auc_score
import pandas as pd


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


def eval_one_epoch_original(hint, tgan, sampler, src, dst, ts, label, val_e_idx_l=None):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    measures_list = []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = val_e_idx_l[s_idx:e_idx] if (val_e_idx_l is not None) else None
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast_original(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut, test=True)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))

            # extra performance measures
            measures_dict = extra_measures(true_label, pred_score)
            measures_list.append(measures_dict)
        measures_df = pd.DataFrame(measures_list)
        avg_measures_dict = measures_df.mean()


    return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc), avg_measures_dict


def eval_one_epoch_modified(hint, tgan, sampler, src, dst, ts, label, val_e_idx_l=None):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    measures_list = []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = val_e_idx_l[s_idx:e_idx] if (val_e_idx_l is not None) else None
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)

            if sampler.neg_sample != 'rnd':
                negative_samples_sources, negative_samples_destinations = \
                    sampler.sample(size, ts_l_cut[0], ts_l_cut[-1])
            else:
                negative_samples_sources, negative_samples_destinations = sampler.sample(size)
                negative_samples_sources = src_l_cut

            pos_prob = tgan.contrast_modified(src_l_cut, dst_l_cut, ts_l_cut, pos_e=True, e_idx_l=e_l_cut, test=True)
            neg_prob = tgan.contrast_modified(negative_samples_sources, negative_samples_destinations, ts_l_cut,
                                              pos_e=False, e_idx_l=None, test=True)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))

            # extra performance measures
            measures_dict = extra_measures(true_label, pred_score)
            measures_list.append(measures_dict)
        measures_df = pd.DataFrame(measures_list)
        avg_measures_dict = measures_df.mean()

    return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc), avg_measures_dict