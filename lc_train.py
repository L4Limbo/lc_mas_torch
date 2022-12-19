import os
import gc
import random
import pprint
from six.moves import range
from markdown2 import markdown
from time import gmtime, strftime
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import lc_options
from lc_dataloader import LCDataset
from torch.utils.data import DataLoader
from eval_utils.lc_rank_answerer import rankABot
from eval_utils.lc_rank_questioner import rankQBot
from utils import lc_utilities as utils
from gensim.models import KeyedVectors
import json

# ---------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------

print('Loading Vocabulary and Vectors')
word2vec = KeyedVectors.load_word2vec_format(
    'data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
vocabulary = json.load(open('data/processed_data/processed_data.json', 'r'))


# Read the command line options
params = lc_options.readCommandLine()
params['trainMode'] = 'rl-full-QAf'
# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
if params['useGPU']:
    torch.cuda.manual_seed_all(params['randomSeed'])

# Setup dataloader
splits = ['train', 'val', 'test']

dataset = LCDataset(params, splits)

# Params to transfer from dataset
transfer = ['vocabSize', 'numOptions', 'numRounds']
for key in transfer:
    if hasattr(dataset, key):
        params[key] = getattr(dataset, key)

# Create save path and checkpoints folder
os.makedirs('checkpoints', exist_ok=True)
os.mkdir(params['savePath'])

# Loading Modules
parameters = []
aBot = None
qBot = None

# Loading A-Bot
if params['trainMode'] in ['sl-abot', 'rl-full-QAf']:
    aBot, loadedParams, optim_state = utils.loadModel(params, 'abot')
    for key in loadedParams:
        params[key] = loadedParams[key]
    parameters.extend(aBot.parameters())

# Loading Q-Bot
if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
    qBot, loadedParams, optim_state = utils.loadModel(params, 'qbot')
    for key in loadedParams:
        params[key] = loadedParams[key]

    if (params['trainMode'] == 'rl-full-QAf') and params['freezeQFeatNet']:
        qBot.freezeFeatNet()
    # Filtering parameters which require a gradient update
    parameters.extend(filter(lambda p: p.requires_grad, qBot.parameters()))
    # parameters.extend(qBot.parameters())

# Setup pytorch dataloader
dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=False,
    num_workers=params['numWorkers'],
    drop_last=True,
    collate_fn=dataset.collate_fn,
    pin_memory=False)

# Setup optimizer
if params['continue']:
    # Continuing from a loaded checkpoint restores the following
    startIterID = params['ckpt_iterid'] + 1  # Iteration ID
    lRate = params['ckpt_lRate']  # Learning rate
    print("Continuing training from iterId[%d]" % startIterID)
else:
    # Beginning training normally, without any checkpoint
    lRate = params['learningRate']
    startIterID = 0

optimizer = optim.Adam(parameters, lr=lRate)
if params['continue']:  # Restoring optimizer state
    print("Restoring optimizer state dict from checkpoint")
    optimizer.load_state_dict(optim_state)
runningLoss = None

mse_criterion = nn.MSELoss(reduce=False)

numIterPerEpoch = dataset.numDataPoints['train'] // params['batchSize']

print('\n%d iter per epoch.' % numIterPerEpoch)

if params['useCurriculum']:
    if params['continue']:
        rlRound = max(0, 4 - (startIterID // numIterPerEpoch))
    else:
        rlRound = params['numRounds'] - 1
else:
    rlRound = 0

# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------


def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch


start_t = timer()

for epochId, idx, batch in batch_iter(dataloader):

    # Keeping track of iterId and epoch
    iterId = startIterID + idx + (epochId * numIterPerEpoch)
    epoch = iterId // numIterPerEpoch
    gc.collect()

    # Moving current batch to GPU, if available
    if dataset.useGPU:
        batch = {key: v.cuda() if hasattr(v, 'cuda')
                 else v for key, v in batch.items()}

    summary = Variable(batch['summary'], requires_grad=False)
    document = Variable(batch['doc'], requires_grad=False)
    documentLens = Variable(batch['doc_len'], requires_grad=False)
    gtQuestions = Variable(batch['ques'], requires_grad=False)
    gtQuesLens = Variable(batch['ques_len'], requires_grad=False)
    gtAnswers = Variable(batch['ans'], requires_grad=False)
    gtAnsLens = Variable(batch['ans_len'], requires_grad=False)
    options = Variable(batch['opt'], requires_grad=False)
    optionLens = Variable(batch['opt_len'], requires_grad=False)
    gtAnsId = Variable(batch['ans_id'], requires_grad=False)

    # Initializing optimizer and losses
    optimizer.zero_grad()
    loss = 0
    qBotLoss = 0
    aBotLoss = 0
    rlLoss = 0
    featLoss = 0
    qBotRLLoss = 0
    aBotRLLoss = 0
    predFeatures = None
    initialGuess = None
    numRounds = params['numRounds']

    if aBot:
        aBot.train(), aBot.reset()
        aBot.observe(-1, document=document,
                     documentLens=documentLens)
    if qBot:
        qBot.train(), qBot.reset()
        qBot.observe(-1, document=document, documentLens=documentLens)

    # Q-Bot summary feature regression ('guessing') only occurs if Q-Bot is present
    if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:

        initialGuess = qBot.predictSummary()

        prevFeatDist = utils.reward(target=summary[0], generated=initialGuess[0][0][1:],
                                    word2vec=word2vec, vocabulary=vocabulary)
        featLoss += prevFeatDist

    # Iterating over dialog rounds
    for round in range(numRounds):

        # Tracking components which require a forward pass
        # A-Bot dialog model
        forwardABot = (params['trainMode'] == 'sl-abot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot dialog model
        forwardQBot = (params['trainMode'] == 'sl-qbot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot feature regression network
        forwardFeatNet = (
            forwardQBot or params['trainMode'] == 'rl-full-QAf')

        # Answerer Forward Pass
        if forwardABot:
            # Observe Ground Truth (GT) question
            aBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            # Observe GT answer for teacher forcing
            aBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])

            ansLogProbs = aBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Answers
            aBotLoss += utils.maskedNll(ansLogProbs,
                                        gtAnswers[:, round].contiguous())

        # Questioner Forward Pass (dialog model)
        if forwardQBot:
            # Observe GT question for teacher forcing
            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            quesLogProbs = qBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Questions
            qBotLoss += utils.maskedNll(quesLogProbs,
                                        gtQuestions[:, round].contiguous())

            # Observe GT answer for updating dialog history
            qBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])

        MAX_FEAT_ROUNDS = 9

        # Questioner feature regression network forward pass
        if forwardFeatNet and round < MAX_FEAT_ROUNDS:
            # Make an summary prediction after each round
            try:
                predFeatures = qBot.predictSummary()
            except:
                pass
            featDist = utils.reward(target=summary[0], generated=predFeatures[0][0][1:],
                                    word2vec=word2vec, vocabulary=vocabulary)
            featLoss += featDist

        # A-Bot and Q-Bot interacting in RL rounds
        if (params['trainMode'] == 'rl-full-QAf') and round >= rlRound:
            # Run one round of conversation
            questions, quesLens = qBot.forwardDecode(inference='sample')
            qBot.observe(round, ques=questions, quesLens=quesLens)
            aBot.observe(round, ques=questions, quesLens=quesLens)
            answers, ansLens = aBot.forwardDecode(inference='sample')
            aBot.observe(round, ans=answers, ansLens=ansLens)
            qBot.observe(round, ans=answers, ansLens=ansLens)

            # Q-Bot makes a guess at the end of each round
            try:
                predFeatures = qBot.predictSummary()
            except:
                print('didnt predict')
                pass

            # Computing reward based on Q-Bot's predicted summary
            featDist = utils.reward(target=summary[0], generated=predFeatures[0][0][1:],
                                    word2vec=word2vec, vocabulary=vocabulary)
            reward = prevFeatDist.detach() - featDist
            prevFeatDist = featDist

            qBotRLLoss = qBot.reinforce(reward)
            if params['rlAbotReward']:
                aBotRLLoss = aBot.reinforce(reward)
            rlLoss += torch.mean(aBotRLLoss)
            rlLoss += torch.mean(qBotRLLoss)

    # Loss coefficients
    rlCoeff = 1
    rlLoss = rlLoss * rlCoeff
    featLoss = featLoss * params['featLossCoeff']
    # Averaging over rounds
    qBotLoss = (params['CELossCoeff'] * qBotLoss) / numRounds
    aBotLoss = (params['CELossCoeff'] * aBotLoss) / numRounds
    featLoss = featLoss / numRounds  # / (numRounds+1)
    rlLoss = rlLoss / numRounds
    # Total loss
    loss = qBotLoss + aBotLoss + rlLoss + featLoss
    loss.backward()
    optimizer.step()

    # Tracking a running average of loss
    if runningLoss is None:
        runningLoss = loss.data.item()
    else:
        runningLoss = 0.95 * runningLoss + 0.05 * loss.data.item()

    # Decay learning rate
    if lRate > params['minLRate']:
        for gId, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
        lRate *= params['lrDecayRate']
        if iterId % 10 == 0:  # Plot learning rate till saturation
            limbo = 0
            # viz.linePlot(iterId, lRate, 'learning rate', 'learning rate')

    # RL Annealing: Every epoch after the first, decrease rlRound
    if iterId % numIterPerEpoch == 0 and iterId > 0:
        if params['trainMode'] == 'rl-full-QAf':
            rlRound = max(1, rlRound - 1)
            print('Using rl starting at round {}'.format(rlRound))

    # Print every now and then
    if iterId % 10 == 0:
        end_t = timer()  # Keeping track of iteration(s) time
        curEpoch = float(iterId) / numIterPerEpoch
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.3g]'
        printFormat += '[lr: %.3g]'
        printInfo = [
            timeStamp, curEpoch, iterId, end_t - start_t, loss.data.item(), lRate
        ]
        start_t = end_t
        print('Losses: ')
        print(printFormat % tuple(printInfo))
        print('aBot loss: %s' % aBotLoss)
        print('qBot loss: %s' % qBotLoss)
        print('rlBot loss: %s' % rlLoss)
        print('featBot loss: %s' % featLoss)
        print('Total loss: %s\n' % loss)

    # TODO: Evaluate every epoch
    if iterId % (numIterPerEpoch // 1) == 0:
        # Keeping track of epochID
        curEpoch = float(iterId) / numIterPerEpoch
        epochId = (1.0 * iterId / numIterPerEpoch) + 1

        # Set eval mode
        if aBot:
            aBot.eval()
        if qBot:
            qBot.eval()

    print('Validations started...')
    # if aBot and 'ques' in batch:
    #     print("aBot Validation:")

    #     rankMetrics = rankABot(
    #         aBot, dataset, 'val', scoringFunction=utils.maskedNll, exampleLimit=25 * params['batchSize'])

    #     for metric, value in rankMetrics.items():
    #         # plot
    #         pass

    #     if 'logProbsMean' in rankMetrics:
    #         logProbsMean = params['CELossCoeff'] * rankMetrics[
    #             'logProbsMean']
    #         # plot

    #         if params['trainMode'] == 'sl-abot':
    #             valLoss = logProbsMean
    #             # plot

    # if qBot:
    #     print("qBot Validation:")
    #     rankMetrics, roundMetrics = rankQBot(
    #         qBot, dataset, 'val', word2vec=word2vec, vocabulary=vocabulary)

    #     for metric, value in rankMetrics.items():
    #         # plot
    #         pass

    #     if 'logProbsMean' in rankMetrics:
    #         logProbsMean = params['CELossCoeff'] * rankMetrics[
    #             'logProbsMean']
    #         # plot

    #     if 'featLossMean' in rankMetrics:
    #         featLossMean = params['featLossCoeff'] * (
    #             rankMetrics['featLossMean'])
    #         # plot

    #     if 'logProbsMean' in rankMetrics and 'featLossMean' in rankMetrics:
    #         if params['trainMode'] == 'sl-qbot':
    #             valLoss = logProbsMean + featLossMean
    #             # plot

    print('Validations finished ...')

    # Save the model after every epoch
    if iterId % numIterPerEpoch == 0:
        params['ckpt_iterid'] = iterId
        params['ckpt_lRate'] = lRate

        if aBot:
            saveFile = os.path.join(params['savePath'],
                                    'abot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(aBot, optimizer, saveFile, params)
        if qBot:
            saveFile = os.path.join(params['savePath'],
                                    'qbot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(qBot, optimizer, saveFile, params)

print('done')
