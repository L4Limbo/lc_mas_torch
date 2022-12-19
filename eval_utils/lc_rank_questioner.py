import sys
import json
import h5py
import numpy as np
from timeit import default_timer as timer

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import options
import lc.lc_metrics as metrics
from utils import lc_utilities as utils
from lc_dataloader import LCDataset
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
import json
from sklearn.metrics.pairwise import pairwise_distances

from six.moves import range


def rankQBot(qBot, dataset, split, exampleLimit=None, verbose=0, vocabulary=None, word2vec=None):
    '''
        Evaluates Q-Bot performance on summary retrieval when it is shown
        ground truth documents, questions and answers. Q-Bot does not
        generate dialog in this setting - it only encodes ground truth
        documents and dialog in order to perform summary retrieval by
        predicting FC-7 summary features after each round of dialog.

        Arguments:
            qBot    : Q-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            exampleLimit : Maximum number of data points to use from
                           the dataset split. If None, all data points.
    '''

    if (word2vec == None or vocabulary == None):
        assert('Vocabulary and Vectors not provided')

    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit
    numBatches = (numExamples - 1) // batchSize + 1
    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn)

    # enumerate all gt features and all predicted features
    gtSummFeatures = []
    # document + dialog rounds
    roundwiseFeaturePreds = [[] for _ in range(numRounds + 1)]
    logProbsAll = [[] for _ in range(numRounds)]
    featLossAll = [[] for _ in range(numRounds + 1)]
    start_t = timer()
    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {
                key: v.cuda()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }
        else:
            batch = {
                key: v.contiguous()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }
        document = Variable(batch['doc'], volatile=True)
        documentLens = Variable(batch['doc_len'], volatile=True)
        gtQuestions = Variable(batch['ques'], volatile=True)
        gtQuesLens = Variable(batch['ques_len'], volatile=True)
        answers = Variable(batch['ans'], volatile=True)
        ansLens = Variable(batch['ans_len'], volatile=True)
        gtFeatures = Variable(batch['summary'], volatile=True)
        qBot.reset()
        qBot.observe(-1, document=document, documentLens=documentLens)
        predFeatures = qBot.predictSummary()
        # Evaluating round 0 feature regression network
        featLoss = utils.reward(target=gtFeatures[0], generated=predFeatures[0][0][1:],
                                word2vec=word2vec, vocabulary=vocabulary)
        featLossAll[0].append(torch.mean(torch.tensor(featLoss)))
        # Keeping round 0 predictions
        roundwiseFeaturePreds[0].append(predFeatures)
        for round in range(numRounds):
            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            qBot.observe(
                round, ans=answers[:, round], ansLens=ansLens[:, round])
            logProbsCurrent = qBot.forward()
            # Evaluating logProbs for cross entropy
            logProbsAll[round].append(
                utils.maskedNll(logProbsCurrent,
                                gtQuestions[:, round].contiguous()))
            predFeatures = qBot.predictSummary()
            # Evaluating feature regression network
            featLoss = utils.reward(target=gtFeatures[0], generated=predFeatures[0][0][1:],
                                    word2vec=word2vec, vocabulary=vocabulary)
            featLossAll[round + 1].append(torch.mean(torch.tensor(featLoss)))
            # Keeping predictions
            roundwiseFeaturePreds[round + 1].append(predFeatures)
        gtSummFeatures.append(gtFeatures)

        end_t = timer()
        delta_t = " Time: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Qbot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")

    gtFeatures = torch.stack(gtSummFeatures, dim=0).data.cpu().numpy()

    rankMetricsRounds = []
    poolSize = len(dataset)

    # Keeping tracking of feature regression loss and CE logprobs
    logProbsAll = [torch.stack(lprobs, dim=0).mean() for lprobs in logProbsAll]
    featLossAll = [torch.stack(floss, dim=0).mean() for floss in featLossAll]
    roundwiseLogProbs = torch.stack(logProbsAll, dim=0).data.cpu().numpy()
    roundwiseFeatLoss = torch.stack(featLossAll, dim=0).data.cpu().numpy()
    logProbsMean = roundwiseLogProbs.mean()
    featLossMean = roundwiseFeatLoss.mean()

    if verbose:
        print("Percentile mean rank (round, mean, low, high)")
    for round in range(numRounds + 1):

        predFeatures = torch.stack(tuple(roundwiseFeaturePreds[round][0][0]),
                                   dim=0).data.cpu().numpy()
        # num_examples x num_examples

        dists = pairwise_distances(
            predFeatures[0][:-1].reshape(-1, 1), gtFeatures[0][0].reshape(-1, 1))
        ranks = []
        for i in range(dists.shape[0]):
            rank = int(np.where(dists[i, :].argsort() == i)[0]) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(ranks)))
        meanRank = ranks.mean()
        se = ranks.std() / np.sqrt(poolSize)
        meanPercRank = 100 * (1 - (meanRank / poolSize))
        percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
        percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
        if verbose:
            print((round, meanPercRank, percRankLow, percRankHigh))
        rankMetrics['percentile'] = meanPercRank
        rankMetrics['featLoss'] = roundwiseFeatLoss[round]
        if round < len(roundwiseLogProbs):
            rankMetrics['logProbs'] = roundwiseLogProbs[round]
        rankMetricsRounds.append(rankMetrics)

    rankMetricsRounds[-1]['logProbsMean'] = logProbsMean
    rankMetricsRounds[-1]['featLossMean'] = featLossMean

    dataset.split = original_split
    return rankMetricsRounds[-1], rankMetricsRounds
