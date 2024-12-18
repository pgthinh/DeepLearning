#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

"""
Evaluation tools adapted from https://github.com/fartashf/vsepp/blob/master/evaluation.py
"""

import numpy as np
import torch
import random
from sentence_transformers import util
from loguru import logger
from tools.file_io import load_pickle_file
from gensim.models.word2vec import Word2Vec
import ot
from tqdm import tqdm
import pylab as pl
import seaborn as sns
from tools.quality_eval import get_t2a_ranking


def setup_seed(seed):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


# evaluation tools
def a2t(audio_embs, cap_embs, return_ranks=False):
    num_audios = int(audio_embs.shape[0] / 5)
    index_list = []

    ranks = np.zeros(num_audios)
    top1 = np.zeros(num_audios)
    mAP10 = np.zeros(num_audios)
    for index in tqdm(range(num_audios)):
        # get query audio
        audio = audio_embs[5 * index].reshape(1, audio_embs.shape[1]) # size of [1, audio_emb]
        d = util.cos_sim(torch.Tensor(audio), torch.Tensor(cap_embs)).squeeze(0).numpy() # size of [1, #captions]
        inds = np.argsort(d)[::-1] # sorting metric scores
        index_list.append(inds[0])

        inds_map = []

        rank = 1e20
        #########################################################################
        # find the best rank among five captions
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
            if tmp < 10:
                inds_map.append(tmp + 1)
        ##########################################################################
        inds_map = np.sort(np.array(inds_map))
        if len(inds_map) != 0:
            mAP10[index] = np.sum((np.arange(1, len(inds_map) + 1) / inds_map)) / 5
        else:
            mAP10[index] = 0.
        ranks[index] = rank
        top1[index] = inds[0]
    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(mAP10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr
    

def t2a(audio_embs, cap_embs, return_ranks=False):
    num_audios = int(audio_embs.shape[0] / 5)
    audios = np.array([audio_embs[i]for i in range(0, audio_embs.shape[0], 5)])

    ranks = np.zeros(5 * num_audios)
    top1 = np.zeros(5 * num_audios)

    for index in tqdm(range(num_audios)):

        # get query captions
        queries = cap_embs[5 * index: 5 * index + 5]
        d = util.cos_sim(torch.Tensor(queries), torch.Tensor(audios)).numpy() # size of [5 queries, #audios]

        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    
    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr


def a2t_ot(audio_embs, cap_embs, M, train_data=False):
    if not train_data:
        audio = [audio_embs[i] for i in range(0, len(audio_embs), 5)]
    else:
        audio = audio_embs

    rank_list = []

    a = torch.ones(len(audio))/len(audio)
    b = torch.ones(len(cap_embs))/len(cap_embs)

    cap_embs = torch.tensor(cap_embs)
    audio = torch.tensor(audio)

    # mean_cap = torch.mean(cap_embs, dim=0)
    # mean_audio = torch.mean(audio, dim=0)
    # gap = torch.pow(mean_cap-mean_audio, 2).sum().sqrt()

    # Mahanalobis distance ####################################
    if len(audio.shape) == 2:
        pairwise_dist = audio.unsqueeze(1).repeat(1,cap_embs.size(0),1) - cap_embs.unsqueeze(0).repeat(audio.size(0), 1,1)
    else:
        pairwise_dist = audio - cap_embs.unsqueeze(0).repeat(audio.size(0), 1,1)
    t_pairwise_dist = pairwise_dist.transpose(1,2)
    M_dist = torch.einsum("ijk,ikj,kk->ij", pairwise_dist.float(), t_pairwise_dist.float(), M.float().cpu())
    M_dist = torch.sqrt(M_dist)
    M_dist = M_dist/M_dist.max()
    ###########################################################

    d = ot.sinkhorn(a,b,M_dist, reg=0.03, numItermax=10).cpu().numpy()

    for index in range(len(audio)):
        inds = np.argsort(d[index])[::-1] # sort an array by index
        inds_map = []
        rank = 1e20
        if not train_data:
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
                if tmp < 10:
                    inds_map.append(tmp + 1)
            rank_list.append(rank)
        else:
            tmp = np.where(inds == index)[0][0]
            if tmp < rank:
                rank = tmp
            if tmp < 10:
                inds_map.append(tmp + 1)
            rank_list.append(rank)
    preds = np.array(rank_list)

    medr = np.floor(np.median(preds)) + 1
    meanr = preds.mean() + 1
    r1 = np.mean(preds < 1)*100
    r5 = np.mean(preds < 5)*100
    r10 = np.mean(preds < 10)*100
    r50 = np.mean(preds < 50)*100
    return r1, r5, r10, r50, medr, meanr


def t2a_ot(audio_embs, cap_embs, M, train_data=False):
    if not train_data:
        audio = [audio_embs[i] for i in range(0, len(audio_embs), 5)]
    else:
        audio = audio_embs
    rank_list = []
    a = torch.ones(len(cap_embs))/len(cap_embs)
    b = torch.ones(len(audio))/len(audio)

    cap_embs = torch.tensor(cap_embs)
    audio = torch.tensor(audio)

    # Mahanalobis distance ####################################
    if len(audio.shape) == 2:
        pairwise_dist = cap_embs.unsqueeze(1).repeat(1,audio.size(0),1) - audio.unsqueeze(0).repeat(cap_embs.size(0), 1,1)
    else:
        pairwise_dist = cap_embs.unsqueeze(1).repeat(1,audio.size(0),1) - audio.permute(1,0,2)
    t_pairwise_dist = pairwise_dist.transpose(1,2)
    M_dist = torch.einsum("ijk,ikj,kk->ij", pairwise_dist.float(), t_pairwise_dist.float(), M.float().cpu())
    M_dist = torch.sqrt(M_dist)
    M_dist = M_dist/M_dist.max()
    ###########################################################

    d = ot.sinkhorn(a,b,M_dist, reg=0.03, numItermax=10).cpu().numpy() #[#cap_embs, #audio]
    rank_d = []
    for index in range(len(audio)):
        if not train_data:
            for i in range(5*index, 5*index+5, 1):
                inds = np.argsort(d[i])[::-1]
                rank = np.where(inds==index)[0][0]
                rank_d.append(inds)
                rank_list.append(rank)
        else:
            inds = np.argsort(d[index])[::-1]
            rank = np.where(inds==index)[0][0]
            rank_d.append(inds)
            rank_list.append(rank)

    preds = np.array(rank_list)
    rank_d = np.array(rank_d)

    medr = np.floor(np.median(preds)) + 1
    meanr = preds.mean() + 1
    r1 = np.mean(preds < 1)*100
    r5 = np.mean(preds < 5)*100
    r10 = np.mean(preds < 10)*100
    r50 = np.mean(preds < 50)*100
    return r1, r5, r10, r50, medr, meanr

def visual_plan(d):
    fig, ax = pl.subplots()
    # matrix = d[:10,:50]
    matrix = d
    # matrix = torch.from_numpy(matrix)
    # matrix = torch.nn.functional.softmax(matrix, dim=-1).detach().numpy()
    
    # index = [i for i in range(0,50,5)]
    # index = [i for i in range(d.shape[0])]
    # matrix = matrix[:, index]
    pl.matshow(matrix[:30,:30])
    # ax = sns.heatmap(matrix)
    pl.xlabel("Caption", fontsize=20)
    pl.ylabel("Audio", fontsize=20)
    pl.savefig("cosine-ot.png")

def visual_true_plan(d):
    true_pi = torch.zeros(d.shape[0], d.shape[1])
    for i in range(d.shape[0]):
        true_pi[i, i*5:i*5+5]=1
        # true_pi[i,i]=1/d.shape[0]
    true_pi = true_pi/(d.shape[0]*d.shape[1])
    true_pi = true_pi.cpu().numpy()

    # matrix = true_pi[:10,:50]
    matrix = true_pi[:30,:30]
    
    # index = [i for i in range(0,50,5)]
    index = [i for i in range(d.shape[0])]
    # matrix = matrix[:, index]
    matrix = true_pi[:30,:30]
    pl.matshow(matrix)
    # pl.colorbar()
    pl.xlabel("Caption",fontsize=20)
    pl.ylabel("Audio", fontsize=20)
    pl.savefig("cosine-true-plan.png")