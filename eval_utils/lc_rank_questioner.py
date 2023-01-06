import sys
import json
import h5py
import numpy as np
from timeit import default_timer as timer

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import lc_options
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
        predicting summary after each round of dialog.

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

    rankMetrics = {
        'logProbsMean': [],
        'summLossMean': [],
        'rouge': [],
    }

    rouge_scores = {
        'r1': [],
        'r2': [],
        'rl': [],
    }

    start_t = timer()
    logProbsAll = [[] for _ in range(numRounds)]
    summLossAll = [[] for _ in range(numRounds)]
    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {
                key: v.cuda() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }
        else:
            batch = {
                key: v.contiguous() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }
        with torch.no_grad():
            summary = Variable(batch['summ'])
            summaryLens = Variable(batch['summ_len'], requires_grad=False)
            document = Variable(batch['doc'])
            documentLens = Variable(batch['doc_len'])
            questions = Variable(batch['ques'])
            quesLens = Variable(batch['ques_len'])
            answers = Variable(batch['ans'])
            ansLens = Variable(batch['ans_len'])
            options = Variable(batch['opt'])
            optionLens = Variable(batch['opt_len'])
            correctOptionInds = Variable(batch['ans_id'])

        qBot.reset()
        qBot.observe(-1, summaryLens=summaryLens, document=document,
                     documentLens=documentLens)

        rankMetrics['rouge'] = []
        for round in range(numRounds):
            logProbsSum = 0
            summLossSum = 0
            r1 = 0
            r2 = 0
            rl = 0
            qBot.observe(
                round,
                ques=questions[:, round],
                quesLens=quesLens[:, round],
                ans=answers[:, round],
                ansLens=ansLens[:, round])
            quesLogProbs = qBot.forward()
            qBotLoss = utils.maskedNll(quesLogProbs,
                                       questions[:, round].contiguous())
            logProbsSum += qBotLoss
            summLogProbs = qBot.forwardSumm(summary=summary)
            prevSummDist = utils.maskedNll(summLogProbs,
                                           summary.contiguous())
            summLossAll.append(torch.mean(prevSummDist))
            summLossSum += prevSummDist

            predSummary = predSummary = qBot.predictSummary()
            for r_score in utils.rouge_scores(target=summary, generated=qBot.predictSummary()[
                    0], word2vec=word2vec, vocabulary=vocabulary):
                r1 += r_score['rouge-1']['f']
                r2 += r_score['rouge-2']['f']
                rl += r_score['rouge-l']['f']

        rouge_scores['r1'].append(r1)
        rouge_scores['r2'].append(r2)
        rouge_scores['rl'].append(rl)
        rankMetrics['logProbsMean'].append(logProbsSum)
        rankMetrics['summLossMean'].append(summLossSum)

    rankMetrics['summLossMean'] = (sum(
        rankMetrics['summLossMean']) / len(rankMetrics['summLossMean'])).detach()
    rankMetrics['logProbsMean'] = (sum(
        rankMetrics['logProbsMean']) / len(rankMetrics['logProbsMean'])).detach()
    rankMetrics['rouge'] = {
        'r1': sum(rouge_scores['r1']) / len(rouge_scores['r1']),
        'r2': sum(rouge_scores['r2']) / len(rouge_scores['r2']),
        'rl': sum(rouge_scores['rl']) / len(rouge_scores['rl']),
    }

    return rankMetrics
