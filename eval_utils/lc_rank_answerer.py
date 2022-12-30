import sys
import json
import h5py
import numpy as np
from timeit import default_timer as timer

import torch
from torch.autograd import Variable

import lc_options as options
import lc.lc_metrics as metrics
from utils import lc_utilities as utils
from lc_dataloader import LCDataset
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import pairwise_distances

from six.moves import range


def rankOptions(options, gtOptions, scores):
    '''Rank a batch of examples against a list of options.'''
    # Compute score of GT options in 'scores'
    gtScores = scores.gather(1, gtOptions.unsqueeze(1))
    # Sort all predicted scores
    sortedScore, _ = torch.sort(scores, 1)
    # In sorted scores, count how many are greater than the GT score
    ranks = torch.sum(sortedScore.gt(gtScores).float(), 1)
    return ranks + 1


def rankABot(aBot, dataset, split, scoringFunction, exampleLimit=None):
    '''
        Evaluate A-Bot performance on ranking answer option when it is
        shown ground truth summary, documents and questions.

        Arguments:
            aBot    : A-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.
            exampleLimit    : Maximum number of data points to use from
                              the dataset split. If None, all data points.
    '''
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

    ranks = []
    logProbsAll = [[] for _ in range(numRounds)]
    start_t = timer()
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
        aBot.reset()
        aBot.observe(-1, summary=summary, summaryLens=summaryLens, document=document,
                     documentLens=documentLens)
        for round in range(numRounds):
            aBot.observe(
                round,
                ques=questions[:, round],
                quesLens=quesLens[:, round],
                ans=answers[:, round],
                ansLens=ansLens[:, round])
            logProbs = aBot.evalOptions(options[:, round],
                                        optionLens[:, round], scoringFunction)
            logProbsCurrent = aBot.forward()
            logProbsAll[round].append(
                scoringFunction(logProbsCurrent,
                                answers[:, round].contiguous()))
            batchRanks = rankOptions(options[:, round],
                                     correctOptionInds[:, round], logProbs)
            ranks.append(batchRanks)

        end_t = timer()
        delta_t = " Rate: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Abot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")
    dataloader = None

    ranks = torch.cat(ranks, 0)
    rankMetrics = metrics.computeMetrics(ranks.cpu())

    logProbsAll = [torch.stack(lprobs, dim=0).mean() for lprobs in logProbsAll]
    roundwiseLogProbs = torch.stack(logProbsAll, dim=0).data.cpu().numpy()
    logProbsMean = roundwiseLogProbs.mean()
    rankMetrics['logProbsMean'] = logProbsMean

    dataset.split = original_split

    return rankMetrics
