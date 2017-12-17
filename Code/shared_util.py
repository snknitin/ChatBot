import sys
import time
import math
import tables
import torch
import pickle
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()
import unicodedata
import re
import string
import json
import collections
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from model import Index



class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1




# Functions to save and load data
def save_hdf5(qa_idxs, filename):
    '''save the processed data into a hdf5 file'''
    print("writing hdf5..")
    f = tables.open_file(filename, 'w')
    filters = tables.Filters(complib='blosc', complevel=5)
    earrays = f.create_earray(f.root, 'sentences', tables.Int16Atom(), shape=(0,), filters=filters)
    indices = f.create_table("/", 'indices', Index, "a table of indices and lengths")
    count = 0
    pos = 0
    for qa in qa_idxs:
        q = qa[0]
        a = qa[1]
        earrays.append(np.array(q))
        earrays.append(np.array(a))
        ind = indices.row
        ind['pos'] = pos
        ind['q_len'] = len(q)
        ind['a_len'] = len(a)
        ind.append()
        pos += len(q) + len(a)
        count += 1
        if count % 1000000 == 0:
            print(count)
            sys.stdout.flush()
            indices.flush()
        elif count % 100000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
    f.close()


def load_hdf5(self, idxfile):
    """read training sentences(list of int array) from a hdf5 file"""
    table = tables.open_file(idxfile)
    data, index = (table.get_node('/sentences'), table.get_node('/indices'))
    data_len = index.shape[0]
    offset = 0
    print("{} entries".format(data_len))
    questions = []
    answers = []
    while offset < data_len:
        pos, q_len, a_len = index[offset]['pos'], index[offset]['q_len'], index[offset]['a_len']
        offset += 1
        questions.append(data[pos:pos + q_len].astype('int64'))
        answers.append(data[pos + q_len:pos + q_len + a_len].astype('int64'))
    table.close()
    return questions, answers




# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# Training
# ========
#
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#




def textproc(s):
    s=s.lower()
    s=s.replace('\'s ','is ')
    s=s.replace('\'re ','are ')
    s=s.replace('\'m ', 'am ')
    s=s.replace('\'ve ', 'have ')
    s=s.replace('\'ll ','will ')
    s=s.replace('n\'t ', 'not ')
    s=s.replace(' wo not',' will not')
    s=s.replace(' ca not',' can not')
    s=re.sub('[\!;-]+','',s)
    s=re.sub('\.+','.',s)
    if s.endswith(' .'):
        s=s[:-2]
    s=re.sub('\s+',' ',s)
    s=s.strip()
    return s

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def sent2indexes(sentence, vocab):
    return [vocab[word] for word in sentence.split(' ')]

def indexes2sent(indexes, ivocab, ignore_tok=-1):
    indexes=filter(lambda i: i!=ignore_tok, indexes)
    return ' '.join([ivocab[idx] for idx in indexes])

SOS_token = 0
EOS_token = 1
UNK_token = 2

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index.keys() else 2 for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(source,target,pair):
    input_variable = variableFromSentence(source, pair[0])
    target_variable = variableFromSentence(target, pair[1])
    return (input_variable, target_variable)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS",2: "UNK"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Convert data into TFRecord, and save the dict (int -> word) to json file
def create_dict(train_file, vocab_size,process_batch_size):
    file = open(train_file, 'r')
    counter = collections.Counter()
    for i, qaline in enumerate(file):
        line = qaline.translate(string.maketrans("", ""), string.punctuation)
        if line == "":
            break
        line = textproc(line)
        words = line.split()
        counter.update(words)
        if i % process_batch_size == 0 and i:
            print(str(i))
    file.close()

    dict = {'UNK': 2, '<SOS>': 1, '<EOS>': 0}
    count = counter.most_common(vocab_size - 3)  # minus 1 for UNK
    for word, _ in count:
        if word == '':
            continue
        dict[word] = len(dict)
    return dict

def load_dict(filename):
    return json.loads(open(filename, "r").readline())

def savepickle(filepath,value):
    source_file = open(filepath, "wb")
    pickle.dump(value, source_file)
    source_file.close()

def loadpickle(filepath):
    pickle_in = open(filepath, "rb")
    loaded = pickle.load(pickle_in)
    return loaded

def binarize(load_path, save_path, vocab,process_batch_size):
    print("binarizing..")
    load_file = open(load_path, "r")
    qa_idxs = []
    for i, qa_sent in enumerate(load_file):
        if i % process_batch_size == 0 and i:
            print(str(i))
        line = qa_sent.translate(string.maketrans("", ""), string.punctuation)
        if line == "":
            break
        line = line.strip("\r\n").split("\t")
        question = textproc(line[0]).split() + ["<EOS>"]
        answer = textproc(line[1]).split() + ["<EOS>"]
        q_idx = [vocab.get(word, vocab['UNK']) for word in question]
        a_idx = [vocab.get(word, vocab['UNK']) for word in answer]
        qa_idxs.append([q_idx, a_idx])

    load_file.close()
    save_hdf5(qa_idxs, save_path)

######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    fig.savefig('./plot.png',bbox_inches='tight')
    plt.close(fig)

######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#

# output_words, attentions = evaluate(
#     encoder1, attn_decoder1, "je suis trop froid .")
# plt.matshow(attentions.numpy())


######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    fig.savefig('./attention.png',bbox_inches='tight')
    plt.close(fig)



