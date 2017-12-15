from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import json
import re
import random
import pickle


SOS_token = 0
EOS_token = 1


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


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

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

######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readLangs(filename):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(filename,mode="r", encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[textproc(normalizeString(s)) for s in l.strip('\r\n').split('\t')] for l in lines]


    source = Lang("source")
    target = Lang("target")

    return source, target, pairs


######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

MAX_LENGTH = 20

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(filename):
    input_lang, output_lang, pairs = readLangs(filename)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs




if __name__ == "__main__":
    data_path = "E:\GIT_ROOT\Pytorch-Deep-Learning-models\LatentVarseq2seq\data\Cornell\\"
    train_file_in = data_path + "train.txt"
    valid_file_in = data_path + "valid.txt"
    test_file_in = data_path + "test.txt"

    print("Creating dictionary for train...")
    print("Processing Training data...")
    source_path = data_path + "train_source.pickle"
    target_path = data_path + "train_target.pickle"
    pairs_path = data_path + "train_pairs.pickle"

    source, target, pairs = prepareData(train_file_in)
    print(random.choice(pairs))
    print(random.choice(pairs))
    print("Pickling train resource")

    source_file = open(source_path, "wb")
    pickle.dump(source, source_file,protocol=2)
    source_file.close()

    target_file = open(target_path, "wb")
    pickle.dump(target, target_file,protocol=2)
    target_file.close()

    pair_file = open(pairs_path, "wb")
    pickle.dump(pairs, pair_file,protocol=2)
    pair_file.close()



    print("Creating dictionary for val...")
    print("Processing Validation data...")
    val_source_path = data_path + "val_source.pickle"
    val_target_path = data_path + "val_target.pickle"
    val_pairs_path = data_path + "val_pairs.pickle"
    VAL_source, VAL_target, VAL_pairs = prepareData(valid_file_in)
    print("Pickling valid resource")
    source_file = open(val_source_path, "wb")
    pickle.dump(VAL_source, source_file,protocol=2)
    source_file.close()

    target_file = open(val_target_path, "wb")
    pickle.dump(VAL_target, target_file,protocol=2)
    target_file.close()

    pair_file = open(val_pairs_path, "wb")
    pickle.dump(VAL_pairs, pair_file,protocol=2)
    pair_file.close()



