from torchtext.data.metrics import bleu_score
from model2 import *
import os
import shutil
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"

import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

def pad_sentences(sentence):
  s = sentence.split(' ')
  while len(s)<=MAX_LENGTH:
    s.append('[PAD]')
  print(s)
  return s
  

def get_bleu_score(model,test_inputs, tokenizer, vocab):
    predicted = model.generate(test_inputs[0], tokenizer, vocab)
    actual = tokenizer.decode(list((test_inputs[1])))
    predicted = pad_sentences(predicted)
    actual = pad_sentences(actual)
    # print('predicted sentence:', predicted)
    # print('actual sentence:', actual)
    return bleu_score(predicted, actual), ' '.join(predicted), ' '.join(actual)

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout=0.1):
        '''
        Illustrative decoder
        '''
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embedding_size,
                                      )

        self.rnn = nn.GRU(embedding_size, hidden_size, bidirectional=True, dropout=dropout, batch_first=False)
        self.dropout_rate = dropout
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, not_used):
        embedded = self.embedding(input).transpose(0, 1)  # [B,1] -> [ 1, B, D]
        embedded = F.dropout(embedded, self.dropout_rate)

        output = embedded

        output, hidden = self.rnn(output, hidden)

        out = self.out(output.squeeze(0))
        output = F.log_softmax(out, dim=1)
        return output, hidden


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


# decoder = DecoderRNN()


def beam_decode(target_tensor, decoder_hiddens, decoder, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    # target_tensor.to("cpu")
    # decoder_hiddens.to("cpu")
    # encoder_outputs.to("cpu")
    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.tensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,  torch.squeeze(encoder_output, dim=1))

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch
