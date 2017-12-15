from __future__ import unicode_literals, print_function, division
from io import open
import os
import shared_util as su
from loader import LoadDataset,sortbatch
import random
import time
import math
import numpy as np
from model import EncoderRNN, AttnDecoderRNN
from shared_util import Lang
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
use_cuda = torch.cuda.is_available()

import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("DATA_DIR", help="Choose folder where files are stored",type=str)
# parser.add_argument("mode", help="Choose train or eval",type=str)
# parser.add_argument("batch_size", help="Choose batch size",type=int) #300000
# parser.add_argument("max_seq_len", help="Choose max length of sequence",type=int) #20
# parser.add_argument("teacher_forcing_ratio", help="Choose train or val",type=float) #0.5
# parser.add_argument("mode", help="Choose train or val",type=str)
# parser.add_argument("mode", help="Choose train or val",type=str)
#
# args = parser.parse_args()
teacher_forcing_ratio=0.5
SOS_token = 0
EOS_token = 1
UNK_token = 2
max_seq_len=20


def trainIters(encoder, decoder, n_iters, save_every=5000, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    print_time_start = start
    plot_losses = []
    print_loss_total, print_loss_kl, print_loss_decoder  = 0., 0., 0.  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [su.variablesFromPair(source,target,random.choice(pairs))
                      for i in range(n_iters)]
    criterion1 = nn.NLLLoss(weight=None, size_average=True)
    criterion2 = nn.KLDivLoss()
    #criterion3 = nn.PoissonNLLLoss(log_input=True, full=False, size_average=True, eps=1e-08)

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
        kl_anneal_weight = (math.tanh((iter - 3500) / 1000) + 1) / 2
        total_loss,kl_loss,decoder_loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer,kl_anneal_weight, criterion1,criterion2)
        print_loss_total += total_loss
        print_loss_kl += kl_loss
        print_loss_decoder += decoder_loss
        #print_loss_poisson += poisson_loss
        plot_loss_total += total_loss

        if iter % print_every == 0:
            print_loss_total = print_loss_total / print_every
            print_loss_kl = print_loss_kl / print_every
            print_loss_decoder = print_loss_decoder / print_every
            #print_loss_poisson = print_loss_poisson / print_every
            print_time = time.time() - print_time_start
            print_time_start = time.time()
            print('iter %d/%d  step_time:%ds  total_time:%s tol_loss: %.4f kl_loss: %.4f  dec_loss: %.4f ' % (iter, n_iters,print_time,su.timeSince( start, iter / n_iters),print_loss_total,
                     print_loss_kl,print_loss_decoder))
            print_loss_total, print_loss_kl, print_loss_decoder = 0, 0, 0
        if iter % save_every ==0:
            if not os.path.exists('%sseqad_%s/' % (model_dir, str(iter))):
                os.makedirs('%sseqad_%s/' % (model_dir, str(iter)))
            torch.save(f='%sseqad_%s/encoder.pckl' % (model_dir,str(iter)),obj=encoder)
            torch.save(f='%sseqad_%s/decoder.pckl' % (model_dir,str(iter)),obj=decoder)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    su.savepickle(model_dir+"plot_losses.pickle",plot_losses)
    #su.showPlot(plot_losses)


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,kl_anneal_weight,
          criterion1,criterion2,max_length= max_seq_len):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            decoder_loss = criterion1(decoder_output, target_variable[di])
            kl_loss = criterion2(decoder_output, target_variable[di])
            loss = 69*decoder_loss + kl_loss*kl_anneal_weight
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_loss = criterion1(decoder_output, target_variable[di])
            kl_loss = criterion2(decoder_output, target_variable[di])
            loss= 69*decoder_loss+kl_loss*kl_anneal_weight
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def evaluate(encoder, decoder, sentence, vocab, max_length=max_seq_len):
    input_variable = su.variableFromSentence(source,target,sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(target.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=20):
    with open(input_dir + "trainsamples.txt", "w") as target:
        for i in range(n):
            pair = random.choice(pairs)
            target.write('> '+ pair[0])
            target.write('= '+ pair[1])
            output_words, attentions = evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            target.write('< '+ output_sentence)
            target.write('\n\n')
        target.close()

def writeresults(encoder, decoder):
    with open(input_dir + "valpred.txt", "w") as target:
        with open(input_dir + "valid.txt", "w") as doc:
            lines=doc.read().strip().split('\n')
            valid_pairs = [[su.textproc(su.normalizeString(s)) for s in l.strip('\r\n').split('\t')] for l in lines]
            for i in range(len(valid_pairs)):
                qapair = valid_pairs[i]
                target.write('> ' + qapair[0])
                target.write('= ' + qapair[1])
                output_words, attentions = evaluate(encoder, decoder, qapair[0])
                output_sentence = ' '.join(output_words)
                target.write('< ' + output_sentence)
                target.write('\n\n')
            doc.close()
        target.close()



if __name__ == '__main__':
    input_dir = "/mnt/nfs/scratch1/nsamala/dialogsystems/ChatBot-Text-Summarizer/datasets/Cornell/"
    model_dir = input_dir + 'models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # vocab = su.load_dict(input_dir + 'vocab.json')
    # ivocab = {v: k for k, v in vocab.items()}
    source_path = input_dir + "train_source.pickle"
    target_path = input_dir + "train_target.pickle"
    pairs_path = input_dir + "train_pairs.pickle"
    hidden_size = 256

    n_iters = 12000
    print_every = 1000
    save_every = 5000
    plot_every = 100
    learning_rate=0.001
    reload_from = -1

    source=su.loadpickle(source_path)
    target=su.loadpickle(target_path)
    pairs=su.loadpickle(pairs_path)
    encoder = EncoderRNN(source.n_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, target.n_words, 1, dropout_p=0.1)

    if reload_from > 0:  # if using from previous data
        encoder = torch.load(f='%sseqad_%s/encoder.pckl' % (model_dir, str(reload_from)))
        decoder = torch.load(f='%sseqad_%s/decoder.pckl' % (model_dir, str(reload_from)))

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print("Training")
    trainIters(encoder, decoder, n_iters, save_every=save_every, print_every=print_every, plot_every=plot_every, learning_rate=learning_rate)

    evaluateRandomly(encoder, decoder)
    writeresults(encoder, decoder)
