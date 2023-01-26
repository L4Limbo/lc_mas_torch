from eval_utils.lc_rank_questioner import rankQBot
from gensim import similarities
import lc_plotter
import os
import gc
import random
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
import numpy as np

from time import gmtime, strftime
from datetime import datetime


# ---------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------

# Read the command line options
params = lc_options.readCommandLine()

timeStamp = strftime('%d-%b-%y-%X-%a')

print('Loading Vocabulary and Vectors')
vocabulary = json.load(open(params['inputJson'], 'r'))
word2vec = KeyedVectors.load_word2vec_format(
    params['word2vec'], binary=True)

# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
if params['useGPU']:
    torch.cuda.manual_seed_all(params['randomSeed'])

# Setup dataloader
splits = ['train', 'val']
dataset = LCDataset(params, splits)

# Params to transfer from dataset
transfer = ['vocabSize', 'numOptions', 'numRounds']
for key in transfer:
    if hasattr(dataset, key):
        params[key] = getattr(dataset, key)

# Create save path and checkpoints folder
os.makedirs('checkpoints', exist_ok=True)
os.mkdir(params['savePath'])

# Create plot data path and folder
tmstp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
path = './plot_data/json_%s' % tmstp
os.makedirs(path,  exist_ok=True)

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

    # Filtering parameters which require a gradient update
    parameters.extend(filter(lambda p: p.requires_grad, qBot.parameters()))

# Setup pytorch dataloader
dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=True,
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

numIterPerEpoch = dataset.numDataPoints['train'] // params['batchSize']

print('\n%d iter per epoch.' % numIterPerEpoch)

if params['useCurriculum']:
    if params['continue']:
        rlRound = max(0, 9 - (startIterID // numIterPerEpoch))
    else:
        rlRound = params['numRounds'] - 1
else:
    rlRound = 0


# Set up files to save for visualization

train_vis = {
    'iterIds': [],
    'aBotLoss': [],
    'qBotLoss': [],
    'rlLoss': [],
    'summLoss': [],
    'qBotRLLoss': [],
    'sumGenRLLoss':[],
    'aBotRLLoss':[],
    'loss': [],
    'runningLoss': [],
    'reward': [],
    'learning_rate': [lRate]
}

abot_vis_val = {
    'iterIds': [],
    'r1': [],
    'r5': [],
    'r10': [],
    'mean': [],
    'mrr': [],
    'logProbsMean': [],
}

qbot_vis_val = {
    'iterIds': [],
    'logProbsMean': [],
    'summLossMean':[],
    'rouge1': [],
    'rouge2': [],
    'rougel': [],
    'word2vec':[],
    'leven':[],
}

# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------


def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch


start_t = timer()
print(params['simFunction'])
started_time = timer()

for epochId, idx, batch in batch_iter(dataloader):

    # Keeping track of iterId and epoch
    iterId = startIterID + idx + (epochId * numIterPerEpoch)
    epoch = iterId // numIterPerEpoch
    gc.collect()

    # Moving current batch to GPU, if available
    if dataset.useGPU:
        batch = {key: v.cuda() if hasattr(v, 'cuda')
                 else v for key, v in batch.items()}

    summary = Variable(batch['summ'], requires_grad=False)
    summaryLens = Variable(batch['summ_len'], requires_grad=False)
    document = Variable(batch['doc'], requires_grad=False)
    documentLens = Variable(batch['doc_len'], requires_grad=False)
    gtQuestions = Variable(batch['ques'], requires_grad=False)
    gtQuesLens = Variable(batch['ques_len'], requires_grad=False)
    gtAnswers = Variable(batch['ans'], requires_grad=False)
    gtAnsLens = Variable(batch['ans_len'], requires_grad=False)
    options = Variable(batch['opt'], requires_grad=False)
    optionLens = Variable(batch['opt_len'], requires_grad=False)
    gtAnsId = Variable(batch['ans_id'], requires_grad=False)
    
    if params['useGPU']:
        reward = Variable(torch.empty(params['batchSize']), requires_grad=False).cuda()
    else:
        reward = Variable(torch.empty(params['batchSize']), requires_grad=False)

    # Initializing optimizer and losses
    optimizer.zero_grad()
    loss = 0
    qBotLoss = 0
    aBotLoss = 0
    rlLoss = 0
    reward = 0
    summLoss = 0
    qBotRLLoss = 0
    sumGenRLLoss = 0
    aBotRLLoss = 0
    predSummary = None
    initSummary = None
    numRounds = params['numRounds']
    # numRounds = 1 # Override for debugging lesser rounds of dialog

    # Setting training modes for both bots and observing documents, summaries where needed
    if aBot:
        aBot.train(), aBot.reset()
        aBot.observe(-1, summary=summary, summaryLens=summaryLens, document=document,
                     documentLens=documentLens)

    if qBot:
        qBot.train(), qBot.reset()
        qBot.observe(-1, document=document,
                     documentLens=documentLens)

    # Q-Bot summary generation only occurs if Q-Bot is present
    if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
       
        predSummary = qBot.predictSummary()
        summLogProbs = qBot.forwardSumm(summary=summary)
        prevSummDist = utils.maskedNll(summLogProbs,
                                       summary.contiguous())
        summLoss += torch.mean(prevSummDist)
        prevSummDist = torch.mean(prevSummDist)
        
        predGreedySummary = qBot.predictSummary(inference='greedy')
        
        if params['trainMode'] == 'rl-full-QAf':
            if params['simFunction'] == 'self_critic':
                prev_sim = utils.calculate_similarity(target=summary, generated=predSummary[
                                                    0], word2vec=word2vec, vocabulary=vocabulary, sim_type='rouge_comb')
                
                prev_sim_greedy = utils.calculate_similarity(target=summary, generated=predGreedySummary[
                                                    0], word2vec=word2vec, vocabulary=vocabulary, sim_type='rouge_comb')
            
            else:
                prev_sim = utils.calculate_similarity(target=summary, generated=predSummary[
                                                    0], word2vec=word2vec, vocabulary=vocabulary, sim_type=params['simFunction'])
                

    # Iterating over dialog rounds
    for round in range(numRounds):
        '''
        Loop over rounds of dialog. Currently three modes of training are
        supported:

            sl-abot :
                Supervised pre-training of A-Bot model using cross
                entropy loss with ground truth answers

            sl-qbot :
                Supervised pre-training of Q-Bot model using cross
                entropy loss with ground truth questions for the
                dialog model and x error loss for summary generation

            rl-full-QAf :
                RL-finetuning of A-Bot and Q-Bot in a cooperative
                setting where the common reward is the difference
                x.

                Annealing: In order to ease in the RL objective,
                fine-tuning starts with first N-1 rounds of SL
                objective and last round of RL objective - the
                number of RL rounds are increased by 1 after
                every epoch until only RL objective is used for
                all rounds of dialog.

        '''
        # Tracking components which require a forward pass
        # A-Bot dialog model
        forwardABot = (params['trainMode'] == 'sl-abot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot dialog model
        forwardQBot = (params['trainMode'] == 'sl-qbot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot Summary Generation Net
        summGenerationNet = (
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

        # In order to stay true to the original implementation, the feature
        # regression network makes predictions before dialog begins and for
        # the first 9 rounds of dialog.
        MAX_FEAT_ROUNDS = 9

        # Questioner feature regression network forward pass
        if summGenerationNet and round < MAX_FEAT_ROUNDS:
            # Make an summary prediction after each round
            predSummary = qBot.predictSummary()
            predGreedySummary = qBot.predictSummary(inference='greedy')

            summLogProbs = qBot.forwardSumm(summary=summary)
            summDist = utils.maskedNll(summLogProbs,
                                       summary.contiguous())
            summDist = torch.mean(summDist)
            summLoss += summDist
            
            if params['trainMode'] == 'rl-full-QAf':
                if params['simFunction'] == 'self_critic':
                    sim = utils.calculate_similarity(target=summary, generated=predSummary[
                                                        0], word2vec=word2vec, vocabulary=vocabulary, sim_type='rouge_comb')
                    
                    sim_greedy = utils.calculate_similarity(target=summary, generated=predGreedySummary[
                                                        0], word2vec=word2vec, vocabulary=vocabulary, sim_type='rouge_comb')
                
                else:
                    sim = utils.calculate_similarity(target=summary, generated=predSummary[
                                                        0], word2vec=word2vec, vocabulary=vocabulary, sim_type=params['simFunction'])

        # A-Bot and Q-Bot interacting in RL rounds
        if (params['trainMode'] == 'rl-full-QAf') and round >= rlRound:
            # Run one round of conversation
            questions, quesLens = qBot.forwardDecode(inference='sample')
            qBot.observe(round, ques=questions, quesLens=quesLens)
            aBot.observe(round, ques=questions, quesLens=quesLens)
            answers, ansLens = aBot.forwardDecode(inference='sample')
            aBot.observe(round, ans=answers, ansLens=ansLens)
            qBot.observe(round, ans=answers, ansLens=ansLens)

            # Q-Bot generates summary at the end of each round
            # TODO LIMBO: REWARD FUNCTION
            predSummary = qBot.predictSummary()
            predGreedySummary = qBot.predictSummary(inference='greedy')

            summLogProbs = qBot.forwardSumm(summary=summary)
            summDist = utils.maskedNll(summLogProbs,
                                       summary.contiguous())
            summDist = torch.mean(summDist)

            if params['trainMode'] == 'rl-full-QAf':
                if params['simFunction'] == 'self_critic':
                    sim = utils.calculate_similarity(target=summary, generated=predSummary[
                                                        0], word2vec=word2vec, vocabulary=vocabulary, sim_type='rouge_comb')
                    
                    sim_greedy = utils.calculate_similarity(target=summary, generated=predGreedySummary[
                                                        0], word2vec=word2vec, vocabulary=vocabulary, sim_type='rouge_comb')
                
                else:
                    sim = utils.calculate_similarity(target=summary, generated=predSummary[
                                                        0], word2vec=word2vec, vocabulary=vocabulary, sim_type=params['simFunction'])
            
            # Reward Calculation
            reward = sim - prev_sim
            prev_sim = sim

            if params['simFunction'] == 'self_critic': 
                reward_greedy = sim_greedy - prev_sim_greedy
                reward_greedy = reward_greedy
                prev_sim_greedy = sim_greedy
                
                reward = reward - reward_greedy

            if params['useGPU']:
                reward = reward.cuda()

            qBotRLLoss = qBot.reinforce(reward)
            sumGenRLLoss = qBot.reinforceSumm(reward)
            if params['rlAbotReward']:
                aBotRLLoss = aBot.reinforce(reward)

            rlLoss += torch.mean(aBotRLLoss)
            rlLoss += torch.mean(qBotRLLoss)
            rlLoss += torch.mean(sumGenRLLoss)

    # Loss coefficients
    rlCoeff = 10
    rlLoss = rlLoss * rlCoeff
    summLoss = summLoss * params['summLossCoeff']
    # Averaging over rounds
    qBotLoss = (qBotLoss) / numRounds
    aBotLoss = (aBotLoss) / numRounds
    summLoss = summLoss / numRounds
    rlLoss = rlLoss / numRounds
    # Total loss
    loss = qBotLoss + aBotLoss + rlLoss + summLoss
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
        if iterId % 10 == 0:
            train_vis['learning_rate'].append(lRate)

    # RL Annealing: Every epoch after the first, decrease rlRound
    if iterId % numIterPerEpoch == 0 and iterId > 0:
        if params['trainMode'] == 'rl-full-QAf':
            rlRound = max(1, rlRound - 1)
            print('Using rl starting at round {}'.format(rlRound))

    # Print every now and then
    if iterId % 20 == 0:
        end_t = timer()
        curEpoch = float(iterId) / numIterPerEpoch
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.3g]'
        printFormat += '[lr: %.3g]'
        printInfo = [
            timeStamp, curEpoch, iterId, end_t - start_t, loss.data.item(), lRate
        ]
        start_t = end_t
        # plots
        print(printFormat % tuple(printInfo))
        
        if params['trainMode'] == 'rl-full-QAf':
            train_vis['reward'].append(float(torch.mean(reward)))
        train_vis['iterIds'].append(iterId)
        train_vis['aBotLoss'].append(float(aBotLoss))
        train_vis['qBotLoss'].append(float(qBotLoss))
        train_vis['rlLoss'].append(float(rlLoss))
        train_vis['summLoss'].append(float(summLoss))
        try:
            train_vis['qBotRLLoss'].append(float(qBotRLLoss))
            train_vis['sumGenRLLoss'].append(float(sumGenRLLoss))
            train_vis['aBotRLLoss'].append(float(aBotRLLoss))
        except:
            train_vis['qBotRLLoss'].append(float(torch.mean(qBotRLLoss)))
            train_vis['sumGenRLLoss'].append(float(torch.mean(sumGenRLLoss)))
            train_vis['aBotRLLoss'].append(float(torch.mean(aBotRLLoss)))

        train_vis['loss'].append(float(loss))
        train_vis['runningLoss'].append(float(runningLoss))
        

    # TODO: Evaluate every epoch
    # print('Epoch validations ...')
    if iterId % (numIterPerEpoch // 1) == 0:
        # Keeping track of epochID
        curEpoch = float(iterId) / numIterPerEpoch
        epochId = (1.0 * iterId / numIterPerEpoch) + 1

        # Set eval mode
        if aBot:
            aBot.eval()
        if qBot:
            qBot.eval()

        if aBot and 'ques' in batch:
            print("aBot Validation:")
            abot_vis_val['iterIds'].append(iterId)
            rankMetrics = rankABot(
                aBot, dataset, 'val', scoringFunction=utils.maskedNll, exampleLimit=32 * params['batchSize'])

            try:
                for metric, value in rankMetrics.items():
                    try:
                        abot_vis_val[metric].append(value.astype(float))
                    except:
                        pass
            except:
                    pass
                
        if qBot:
            print("qBot Validation:")
            rankMetrics = rankQBot(qBot, dataset, 'val')
            qbot_vis_val['iterIds'].append(iterId)
            try:
                for metric, value in rankMetrics.items():
                    try:
                        qbot_vis_val[metric].append(value.astype(float))
                    except:
                        pass
            except:
                    pass
              

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

            print('Saving data for plots ... ')

        try:
            with open('%s/train_vis.json' % path, 'w') as jsonFile:
                jsonFile.write(json.dumps(train_vis, indent=4))
        except:
            print('didnt save train_vis')

        try:
            with open('%s/abot_vis_val.json' % path, 'w') as jsonFile:
                jsonFile.write(json.dumps(abot_vis_val, indent=4))
        except:
            print('didnt save abot_vis_val')

        try:
            with open('%s/qbot_vis_val.json' % path, 'w') as jsonFile:
                jsonFile.write(json.dumps(qbot_vis_val, indent=4))
        except:
            print('didnt save abot_vis_val')


print('Training Session finished in %s s' % (timer()-started_time))
print('Creating Plots ... ')
lc_plotter.create_plots('json_%s' % tmstp)

print('Done!')