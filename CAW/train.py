import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from eval import *
import logging
import time

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True



def train_val(train_val_data, model, mode, bs, epochs, criterion, optimizer, early_stopper, ngh_finders, rand_samplers,
              logger):
    # unpack the data, prepare for the training
    train_data, val_data = train_val_data
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data
    train_rand_sampler, val_rand_sampler = rand_samplers
    partial_ngh_finder, full_ngh_finder = ngh_finders
    if mode == 't':  # transductive
        model.update_ngh_finder(full_ngh_finder)
    elif mode == 'i':  # inductive
        model.update_ngh_finder(partial_ngh_finder)
    else:
        raise ValueError('training mode {} not found.'.format(mode))
    device = model.n_feat_th.data.device
    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / bs)
    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    for epoch in range(epochs):
        start_time_epoch = time.time()
        acc, ap, auc_roc, m_loss = [], [], [], []
        np.random.shuffle(idx_list)  # shuffle the training samples for every epoch
        logger.info('start {} epoch'.format(epoch))
        # for k in tqdm(range(num_batch)):  # this explodes the log file!!!
        for k in range(num_batch):
            # generate training mini-batch
            s_idx = k * bs
            e_idx = min(num_instance - 1, s_idx + bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx]
            ts_l_cut = train_ts_l[batch_idx]
            e_l_cut = train_e_idx_l[batch_idx]
            label_l_cut = train_label_l[batch_idx]  # currently useless since we are not predicting edge labels
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

            # feed in the data and learn from error
            optimizer.zero_grad()
            model.train()
            pos_prob, neg_prob = model.contrast_original(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut,
                                                         e_l_cut)  # the core training code
            pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
            neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False)
            loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
            loss.backward()
            optimizer.step()

            # collect training results
            with torch.no_grad():
                model.eval()
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                # f1.append(f1_score(true_label, pred_label))
                m_loss.append(loss.item())
                auc_roc.append(roc_auc_score(true_label, pred_score))

        # validation phase use all information
        val_acc, val_ap, val_f1, val_auc, avg_measures_dict = eval_one_epoch_original('val for {} nodes'.format(mode), model, val_rand_sampler, val_src_l,
                                                                                      val_dst_l, val_ts_l, val_label_l, val_e_idx_l)
        logger.info('epoch: {}:'.format(epoch))
        logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
        logger.info('train auc: {}, val auc: {}'.format(np.mean(auc_roc), val_auc))
        logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))
        if epoch == 0:
            # save things for data analysis
            checkpoint_dir = '/'.join(model.get_checkpoint_path(0).split('/')[:-1])
            model.ngh_finder.save_ngh_stats(checkpoint_dir)  # for data analysis
            model.save_common_node_percentages(checkpoint_dir)

        # early stop check and checkpoint saving
        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), model.get_checkpoint_path(epoch))
