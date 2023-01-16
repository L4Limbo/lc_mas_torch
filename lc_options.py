import os
import argparse
from time import gmtime, strftime


def readCommandLine(argv=None):
    parser = argparse.ArgumentParser(
        description='Train and Test the model')

    # -------------------------------------------------------------------------
    # Data input settings
    parser.add_argument('-inputSumm', default='data/processed_data/summ.json',
                        help='json file with summaries')
    parser.add_argument('-inputQues', default='data/processed_data/processed_data.h5',
                        help='HDF5 file with preprocessed questions')
    parser.add_argument('-inputJson', default='data/processed_data/processed_data.json',
                        help='JSON file with info and vocab')
    parser.add_argument('-word2vec', default='data/word2vec/GoogleNews-vectors-negative300.bin',
                        help='.bin file with word2vec pretrained model')
    # -------------------------------------------------------------------------
    # Logging settings
    parser.add_argument('-verbose', type=int, default=1,
                        help='Level of verbosity (default 1 prints some info)',
                        choices=[1, 2])
    parser.add_argument('-savePath', default='checkpoints/',
                        help='Path to save checkpoints')
    parser.add_argument('-saveName', default='',
                        help='Name of save directory within savePath')
    parser.add_argument('-startFrom', type=str, default='',
                        help='Copy weights from model at this path')
    parser.add_argument('-qstartFrom', type=str, default='',
                        help='Copy weights from qbot model at this path')
    parser.add_argument('-continue', action='store_true',
                        help='Continue training from last epoch')
    # -------------------------------------------------------------------------
    # Model params for both a-bot and q-bot
    parser.add_argument('-randomSeed', default=32, type=int,
                        help='Seed for random number generators')
    parser.add_argument('-summSize', default=60, type=int,
                        help='Size of the generated summary')
    parser.add_argument('-embedSize', default=300, type=int,
                        help='Size of input word embeddings')
    parser.add_argument('-rnnHiddenSize', default=256, type=int,
                        help='Size of the LSTM state')
    parser.add_argument('-numLayers', default=2, type=int,
                        help='Number of layers in LSTM')

    # A-Bot encoder + decoder
    parser.add_argument('-encoder', default='hre-ques-lateim-hist',
                        help='Name of the encoder to use',
                        choices=['hre-ques-lateim-hist'])
    parser.add_argument('-decoder', default='gen',
                        help='Name of the decoder to use (gen)',
                        choices=['gen'])
    # Q-bot encoder + decoder
    parser.add_argument('-qencoder', default='hre-ques-lateim-hist',
                        help='Name of the encoder to use',
                        choices=['hre-ques-lateim-hist'])
    parser.add_argument('-qdecoder', default='gen',
                        help='Name of the decoder to use (only gen supported now)',
                        choices=['gen'])

    # -------------------------------------------------------------------------
    # Optimization / training params
    parser.add_argument('-trainMode', default='rl-full-QAf',
                        help='What should train.py do?',
                        choices=['sl-abot', 'sl-qbot', 'rl-full-QAf'])
    parser.add_argument('-numRounds', default=10, type=int,
                        help='Number of rounds of dialog (max 10)')
    parser.add_argument('-batchSize', default=5, type=int,
                        help='Batch size (number of threads) '
                        '(Adjust base on GPU memory)')
    parser.add_argument('-learningRate', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('-minLRate', default=5e-5, type=float,
                        help='Minimum learning rate')
    parser.add_argument('-dropout', default=0.1, type=float, help='Dropout')
    parser.add_argument('-numEpochs', default=40, type=int, help='Epochs')
    parser.add_argument('-lrDecayRate', default=0.999962372474343, type=float,
                        help='Decay for learning rate')
    parser.add_argument('-CELossCoeff', default=1, type=float,
                        help='Coefficient for cross entropy loss')
    parser.add_argument('-summLossCoeff', default=1, type=float,
                        help='Coefficient for summarization loss')
    parser.add_argument('-useCurriculum', default=1, type=int,
                        help='Use curriculum or for RL training (1) or not (0)')
    parser.add_argument('-rlAbotReward', default=1, type=int,
                        help='Choose whether RL reward goes to A-Bot')

    # Other training environmnet settings
    parser.add_argument('-useGPU', action='store_true', help='Use GPU or CPU')
    parser.add_argument('-numWorkers', default=0, type=int,
                        help='Number of worker threads in dataloader')
    
    # -------------------------------------------------------------------------
    # Reward functions
    parser.add_argument('-simFunction', default='rouge',
                        help='Choose a similarity function for reward in RL finetuning',
                        choices=['rouge', 'levenshtein', 'word2vec', 'self_critic'])

    # -------------------------------------------------------------------------
    # Evaluation params
    parser.add_argument('-beamSize', default=5, type=int,
                        help='Beam width for beam-search sampling')
    parser.add_argument('-evalModeList', default=[], nargs='+',
                        help='What task should the evaluator perform?',
                        choices=['ABotRank', 'QBotRank', 'QABotsRank', 'dialog'])
    parser.add_argument('-evalSplit', default='val',
                        choices=['train', 'val'])
    parser.add_argument('-evalTitle', default='eval',
                        help='If generating a plot, include this in the title')
    # -------------------------------------------------------------------------

    try:
        parsed = vars(parser.parse_args(args=argv))
    except IOError as msg:
        parser.error(str(msg))

    if parsed['saveName']:
        # Custom save file path
        parsed['savePath'] = os.path.join(parsed['savePath'],
                                          parsed['saveName'])
    else:
        # Standard save path with time stamp
        import random
        timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())
        parsed['savePath'] = os.path.join(parsed['savePath'], timeStamp)
        parsed['savePath'] += '_{:0>6d}'.format(random.randint(0, 10e6))

    # check if history is needed
    parsed['useHistory'] = True if 'hist' in parsed['encoder'] else False

    # check if summary is needed
    if 'lateim' in parsed['encoder']:
        parsed['useSumm'] = 'late'
    elif 'summ' in parsed['encoder']:
        parsed['useSumm'] = True
    else:
        parsed['useSumm'] = False
    return parsed
