import math

import pandas as pd
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import torch
from sklearn.metrics import *


def eval_edge_prediction_original(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    measures_list = []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities_original(sources_batch, destinations_batch,
                                                                           negative_samples, timestamps_batch,
                                                                           edge_idxs_batch, n_neighbors)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

            # extra performance measures
            measures_dict = extra_measures(true_label, pred_score)
            measures_list.append(measures_dict)
        measures_df = pd.DataFrame(measures_list)
        avg_measures_dict = measures_df.mean()

    return np.mean(val_ap), np.mean(val_auc), avg_measures_dict


def eval_edge_prediction_modified(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    measures_list = []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)

            if negative_edge_sampler.neg_sample != 'rnd':
                negative_samples_sources, negative_samples_destinations = \
                    negative_edge_sampler.sample(size,
                                                 timestamps_batch[0],
                                                 timestamps_batch[-1])
            else:
                negative_samples_sources, negative_samples_destinations = negative_edge_sampler.sample(size)
                negative_samples_sources = sources_batch

            pos_e = True
            pos_prob = model.compute_edge_probabilities_modified(sources_batch, destinations_batch, timestamps_batch,
                                                                 edge_idxs_batch,
                                                                 pos_e, n_neighbors)
            pos_e = False
            neg_prob = model.compute_edge_probabilities_modified(negative_samples_sources,
                                                                 negative_samples_destinations,
                                                                 timestamps_batch,
                                                                 edge_idxs_batch,
                                                                 pos_e, n_neighbors)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

            # extra performance measures
            measures_dict = extra_measures(true_label, pred_score)
            measures_list.append(measures_dict)
        measures_df = pd.DataFrame(measures_list)
        avg_measures_dict = measures_df.mean()

    return np.mean(val_ap), np.mean(val_auc), avg_measures_dict

def extract_edge_embeddings(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    """
    Convention for saving the embedding is as follows:
        - source id (length: 1)
        - destination id (length: 1)
        - timestamp (length: 1)
        - edge index (length: 1)
        - label (positive: 1; negative: 0) (length: 1)
        - source embedding (length: 172)
        - destination embedding (length: 172)
    """

    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        edge_emb = []
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)

            if negative_edge_sampler.neg_sample != 'rnd':
                negative_samples_sources, negative_samples_destinations = \
                    negative_edge_sampler.sample(size,
                                                 timestamps_batch[0],
                                                 timestamps_batch[-1])
            else:
                negative_samples_sources, negative_samples_destinations = negative_edge_sampler.sample(size)
                negative_samples_sources = sources_batch
            # positive edges
            pos_e = True
            pos_source_node_embedding, pos_destination_node_embedding = model.compute_temporal_embeddings_modified(
                sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, pos_e, n_neighbors)
            edge_lbl = np.ones((size, 1))

            pos_edge_features = np.concatenate([np.asarray(sources_batch).reshape(size, 1),
                                                np.asarray(destinations_batch).reshape(size, 1),
                                                np.asarray(timestamps_batch).reshape(size, 1),
                                                np.asarray(edge_idxs_batch).reshape(size, 1),
                                                edge_lbl, pos_source_node_embedding.cpu().numpy(),
                                                pos_destination_node_embedding.cpu().numpy()], axis=1)
            # edge_emb = np.append(edge_emb, pos_edge_features, axis=0)
            edge_emb.append(pos_edge_features)

            # negative edges
            pos_e = False
            neg_source_node_embedding, neg_destination_node_embedding = model.compute_temporal_embeddings_modified(
                negative_samples_sources, negative_samples_destinations, timestamps_batch, edge_idxs_batch, pos_e, n_neighbors)
            edge_lbl = np.zeros((size, 1))
            neg_edge_features = np.concatenate([np.asarray(negative_samples_sources).reshape(size, 1),
                                                np.asarray(negative_samples_destinations).reshape(size, 1),
                                                np.asarray(timestamps_batch).reshape(size, 1),
                                                np.asarray(edge_idxs_batch).reshape(size, 1),
                                                edge_lbl,
                                                neg_source_node_embedding.cpu().numpy(),
                                                neg_destination_node_embedding.cpu().numpy()], axis=1)
            # edge_emb = np.append(edge_emb, neg_edge_features, axis=0)
            edge_emb.append(neg_edge_features)
        edge_emb = np.concatenate([np.asarray(emb_arr) for emb_arr in edge_emb], axis=0)

    return edge_emb



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
    perf_dict['acc'] = perf_half_dict['acc']
    perf_dict['prec'] = perf_half_dict['prec']
    perf_dict['rec'] = perf_half_dict['rec']
    perf_dict['f1'] = perf_half_dict['f1']

    return perf_dict


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
    pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx: e_idx]
            destinations_batch = data.destinations[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx: e_idx]

            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                         destinations_batch,
                                                                                         destinations_batch,
                                                                                         timestamps_batch,
                                                                                         edge_idxs_batch,
                                                                                         n_neighbors)
            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc
