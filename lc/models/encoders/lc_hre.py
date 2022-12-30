import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import lc_utilities as utils


class Encoder(nn.Module):
    def __init__(self,
                 vocabSize,
                 embedSize,
                 rnnHiddenSize,
                 numLayers,
                 useSumm,
                 numRounds,
                 isAnswerer,
                 dropout=0,
                 startToken=None,
                 endToken=None,
                 **kwargs):
        super(Encoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        assert self.numLayers > 1, "Less than 2 layers not supported!"
        if useSumm:
            self.useSumm = useSumm if useSumm != True else 'early'
        else:
            self.useSumm = False
        self.numRounds = numRounds
        self.dropout = dropout
        self.isAnswerer = isAnswerer
        self.startToken = startToken
        self.endToken = endToken

        # modules
        self.wordEmbed = nn.Embedding(
            self.vocabSize, self.embedSize, padding_idx=0)

        # question encoder
        if self.useSumm == 'early':
            quesInputSize = self.embedSize * 2
            dialogInputSize = 2 * self.rnnHiddenSize
        elif self.useSumm == 'late':
            quesInputSize = self.embedSize
            dialogInputSize = 2 * self.rnnHiddenSize
        elif self.isAnswerer:
            quesInputSize = self.embedSize
            dialogInputSize = 2 * self.rnnHiddenSize
        else:
            dialogInputSize = self.rnnHiddenSize
        if self.isAnswerer:
            self.quesRNN = nn.LSTM(
                quesInputSize,
                self.rnnHiddenSize,
                self.numLayers,
                batch_first=True,
                dropout=0)

        # history encoder
        self.factRNN = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=0)

        # dialog rnn
        self.dialogRNN = nn.LSTMCell(dialogInputSize, self.rnnHiddenSize)

    def reset(self):
        # batchSize is inferred from input
        self.batchSize = 0

        # Input data
        self.summaryTokens = None
        self.summaryEmbed = None
        self.summaryLens = None

        self.documentTokens = None
        self.documentEmbed = None
        self.documentLens = None

        self.questionTokens = []
        self.questionEmbeds = []
        self.questionLens = []

        self.answerTokens = []
        self.answerEmbeds = []
        self.answerLengths = []

        # Hidden embeddings
        self.factEmbeds = []
        self.questionRNNStates = []
        self.dialogRNNInputs = []
        self.dialogHiddens = []

    def _initHidden(self):
        '''Initial dialog rnn state - initialize with zeros'''
        # Dynamic batch size inference
        assert self.batchSize != 0, 'Observe something to infer batch size.'
        someTensor = self.dialogRNN.weight_hh.data
        h = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        c = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        return (Variable(h), Variable(c))

    def observe(self,
                round,
                summary=None,
                summaryLens=None,
                document=None,
                ques=None,
                ans=None,
                documentLens=None,
                quesLens=None,
                ansLens=None):
        '''
        Store dialog input to internal model storage

        Note that all input sequences are assumed to be left-aligned (i.e.
        right-padded). Internally this alignment is changed to right-align
        for ease in computing final time step hidden states of each RNN
        '''

        if summary is not None:
            assert round == -1
            assert summaryLens is not None, "Document lengths required!"
            summary, summaryLens = self.processSequence(
                summary, summaryLens)
            self.summaryTokens = summary
            self.summaryLens = summaryLens
            self.batchSize = len(self.summaryTokens)

        if document is not None:
            assert round == -1
            assert documentLens is not None, "Document lengths required!"
            document, documentLens = self.processSequence(
                document, documentLens)
            self.documentTokens = document
            self.documentLens = documentLens
            self.batchSize = len(self.documentTokens)
        if ques is not None:
            assert round == len(self.questionEmbeds)
            assert quesLens is not None, "Questions lengths required!"
            ques, quesLens = self.processSequence(ques, quesLens)
            self.questionTokens.append(ques)
            self.questionLens.append(quesLens)
        if ans is not None:
            assert round == len(self.answerEmbeds)
            assert ansLens is not None, "Answer lengths required!"
            ans, ansLens = self.processSequence(ans, ansLens)
            self.answerTokens.append(ans)
            self.answerLengths.append(ansLens)

    def processSequence(self, seq, seqLen):
        ''' Strip <START> and <END> token from a left-aligned sequence'''
        return seq[:, 1:], seqLen - 1

    def embedInputDialog(self):
        '''
        Lazy embedding of input:
            Calling observe does not process (embed) any inputs. Since
            self.forward requires embedded inputs, this function lazily
            embeds them so that they are not re-computed upon multiple
            calls to forward in the same round of dialog.
        '''
        # Embed summary, occurs once per dialog
        if self.isAnswerer and self.summaryEmbed is None:
            self.summaryEmbed = self.wordEmbed(self.summaryTokens)
        # Embed document, occurs once per dialog
        if self.documentEmbed is None:
            self.documentEmbed = self.wordEmbed(self.documentTokens)
        # Embed questions
        while len(self.questionEmbeds) < len(self.questionTokens):
            idx = len(self.questionEmbeds)
            self.questionEmbeds.append(
                self.wordEmbed(self.questionTokens[idx]))
        # Embed answers
        while len(self.answerEmbeds) < len(self.answerTokens):
            idx = len(self.answerEmbeds)
            self.answerEmbeds.append(self.wordEmbed(self.answerTokens[idx]))

    def embedFact(self, factIdx):
        '''Embed facts i.e. document and round 0 or question-answer pair otherwise'''
        # Document
        if factIdx == 0:
            if self.useSumm == 'late' and self.isAnswerer:
                seqTokens, seqLens = self.documentTokens, self.documentLens
                summTokens, summLens = self.summaryTokens, self.summaryLens

                seqSummTokens = utils.concatPaddedSequences(
                    summTokens, summLens, seqTokens, seqLens, padding='right')

                seqSumm = self.wordEmbed(seqSummTokens)
                seqSummLens = seqLens + summLens

                seqSummEmbed, states = utils.dynamicRNN(
                    self.factRNN, seqSumm, seqSummLens, returnStates=True)
                factEmbed = seqSummEmbed
            else:
                seq, seqLens = self.documentEmbed, self.documentLens
                factEmbed, states = utils.dynamicRNN(
                    self.factRNN, seq, seqLens, returnStates=True)
        # QA pairs
        elif factIdx > 0:
            quesTokens, quesLens = \
                self.questionTokens[factIdx -
                                    1], self.questionLens[factIdx - 1]
            ansTokens, ansLens = \
                self.answerTokens[factIdx - 1], self.answerLengths[factIdx - 1]

            qaTokens = utils.concatPaddedSequences(
                quesTokens, quesLens, ansTokens, ansLens, padding='right')
            qa = self.wordEmbed(qaTokens)
            qaLens = quesLens + ansLens
            qaEmbed, states = utils.dynamicRNN(
                self.factRNN, qa, qaLens, returnStates=True)
            factEmbed = qaEmbed
        factRNNstates = states
        self.factEmbeds.append((factEmbed, factRNNstates))

    def embedQuestion(self, qIdx):
        '''Embed questions'''
        quesIn = self.questionEmbeds[qIdx]
        quesLens = self.questionLens[qIdx]
        if self.useSumm == 'early':
            summary = self.summaryEmbed.unsqueeze(
                1).repeat(1, quesIn.size(1), 1)
            quesIn = torch.cat([quesIn, summary], 2)
        qEmbed, states = utils.dynamicRNN(
            self.quesRNN, quesIn, quesLens, returnStates=True)
        quesRNNstates = states
        self.questionRNNStates.append((qEmbed, quesRNNstates))

    def concatDialogRNNInput(self, histIdx):
        currIns = [self.factEmbeds[histIdx][0]]
        if self.isAnswerer:
            currIns.append(self.questionRNNStates[histIdx][0])
        hist_t = torch.cat(currIns, -1)
        self.dialogRNNInputs.append(hist_t)

    def embedDialog(self, dialogIdx):
        if dialogIdx == 0:
            hPrev = self._initHidden()
        else:
            hPrev = self.dialogHiddens[-1]
        inpt = self.dialogRNNInputs[dialogIdx]
        hNew = self.dialogRNN(inpt, hPrev)
        self.dialogHiddens.append(hNew)

    def forward(self):
        '''
        Returns:
            A tuple of tensors (H, C) each of shape (batchSize, rnnHiddenSize)
            to be used as the initial Hidden and Cell states of the Decoder.
            See notes at the end on how (H, C) are computed.
        '''

        # Lazily embed input Documents, Questions and Answers
        self.embedInputDialog()

        if self.isAnswerer:
            # For A-Bot, current round is the number of facts present,
            # which is number of questions observed - 1 (as opposed
            # to len(self.answerEmbeds), which may be inaccurate as
            round = len(self.questionEmbeds) - 1
        else:
            # For Q-Bot, current round is the number of facts present,
            # which is same as the number of answers observed
            round = len(self.answerEmbeds)

        # Lazy computation of internal hidden embeddings (hence the while loops)

        # Infer any missing facts
        while len(self.factEmbeds) <= round:
            factIdx = len(self.factEmbeds)
            self.embedFact(factIdx)

        # Embed any un-embedded questions (A-Bot only)
        if self.isAnswerer:
            while len(self.questionRNNStates) <= round:
                qIdx = len(self.questionRNNStates)
                self.embedQuestion(qIdx)

        # Concat facts and/or questions (i.e. history) for input to dialogRNN
        while len(self.dialogRNNInputs) <= round:
            histIdx = len(self.dialogRNNInputs)
            self.concatDialogRNNInput(histIdx)

        # Forward dialogRNN one step
        while len(self.dialogHiddens) <= round:
            dialogIdx = len(self.dialogHiddens)
            self.embedDialog(dialogIdx)

        # Latest dialogRNN hidden state
        dialogHidden = self.dialogHiddens[-1][0]

        '''
        Return hidden (H_link) and cell (C_link) states as per the following rule:
        (Currently this is defined only for numLayers == 2)
        If A-Bot:
          C_link == Question encoding RNN cell state (quesRNN)
          H_link ==
              Layer 0 : Question encoding RNN hidden state (quesRNN)
              Layer 1 : DialogRNN hidden state (dialogRNN)

        If Q-Bot:
            C_link == Fact encoding RNN cell state (factRNN)
            H_link ==
                Layer 0 : Fact encoding RNN hidden state (factRNN)
                Layer 1 : DialogRNN hidden state (dialogRNN)
        '''
        if self.isAnswerer:
            # Latest quesRNN states
            quesRNNstates = self.questionRNNStates[-1][1]
            C_link = quesRNNstates[1]
            H_link = quesRNNstates[0][:-1]
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)
        else:
            factRNNstates = self.factEmbeds[-1][1]  # Latest factRNN states
            C_link = factRNNstates[1]
            H_link = factRNNstates[0][:-1]
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)

        return H_link, C_link
