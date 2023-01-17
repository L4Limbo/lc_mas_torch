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
        generating summary after each round of dialog.

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

    start_t = timer()
    logProbsAll = [[] for _ in range(numRounds)]
    summLossAll = [[] for _ in range(numRounds + 1)]
    
    for idx, batch in enumerate(dataloader):
        print(idx)
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
            summaryLens = Variable(batch['summ_len'])
            document = Variable(batch['doc'])
            documentLens = Variable(batch['doc_len'])
            gtQuestions = Variable(batch['ques'])
            gtQuesLens = Variable(batch['ques_len'] )
            answers = Variable(batch['ans'])
            ansLens = Variable(batch['ans_len'])

        qBot.reset()
        qBot.observe(-1, document=document, documentLens=documentLens)
        
        predSummary = qBot.predictSummary()
        predGreedySummary = qBot.predictSummary(inference='greedy')

        summLogProbs = qBot.forwardSumm(summary=summary)
        summDist = utils.maskedNll(summLogProbs,
                                    summary.contiguous())
        summDist = torch.mean(summDist)
        summLossAll[0].append(summDist)

        for round in range(numRounds):
            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            qBot.observe(
                round, ans=answers[:, round], ansLens=ansLens[:, round])
            
            quesLogProbs = qBot.forward()

            logProbsAll[round].append(utils.maskedNll(quesLogProbs,
                            gtQuestions[:, round].contiguous()))
            
            predSummary = qBot.predictSummary()
            predGreedySummary = qBot.predictSummary(inference='greedy')

            summLogProbs = qBot.forwardSumm(summary=summary)
            summDist = utils.maskedNll(summLogProbs,
                                        summary.contiguous())
            summDist = torch.mean(summDist)
            summLossAll[round + 1].append(summDist)   

        end_t = timer()
        delta_t = " Time: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Qbot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")
    
    

    
    logProbsAll = [torch.stack(lprobs, dim=0).mean() for lprobs in logProbsAll]
    roundwiseLogProbs = torch.stack(logProbsAll, dim=0).data.cpu().numpy()
    logProbsMean = roundwiseLogProbs.mean()

    summLossAll = [torch.stack(lprobs, dim=0).mean() for lprobs in summLossAll]
    roundwiseSummProbs = torch.stack(summLossAll, dim=0).data.cpu().numpy()
    summLossMean = roundwiseSummProbs.mean()
    dataset.split = original_split
    return {
        'logProbsMean' : logProbsMean,
        'summLossMean': summLossMean
    }
