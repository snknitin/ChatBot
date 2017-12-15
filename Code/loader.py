import torch
import torch.utils.data as data
import tables
import numpy as np

def sortbatch(q_batch, a_batch, q_lens, a_lens):
    """
    sort sequences according to their lengthes in descending order
    """
    maxlen_q = max(q_lens)
    maxlen_a = max(a_lens)
    q=q_batch[:,:maxlen_q-1]
    a=a_batch[:,:maxlen_a-1]
    sorted_idx = torch.LongTensor(a_lens.numpy().argsort()[::-1].copy())
    return q[sorted_idx], a[sorted_idx], q_lens[sorted_idx], a_lens[sorted_idx]


class LoadDataset(data.Dataset):
    def __init__(self, filepath, max_seq_len):
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.max_seq_len = max_seq_len

        print("loading data...")
        table = tables.open_file(filepath)
        self.data = table.get_node('/sentences')
        self.index = table.get_node('/indices')
        self.data_len = self.index.shape[0]
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        # print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        pos, q_len, a_len = self.index[offset]['pos'], self.index[offset]['q_len'], self.index[offset]['a_len']
        question = self.data[pos:pos + q_len].astype('int64')
        answer = self.data[pos + q_len:pos + q_len + a_len].astype('int64')

        ## Padding ##
        if len(question) < self.max_seq_len:
            question = np.append(question, [0] * self.max_seq_len)
        question = question[:self.max_seq_len]
        question[-1] = 0
        if len(answer) < self.max_seq_len:
            answer = np.append(answer, [0] * self.max_seq_len)
        answer = answer[:self.max_seq_len]
        answer[-1] = 0

        ## get real seq len
        q_len = min(int(q_len), self.max_seq_len)  # real length of question for training
        a_len = min(int(a_len), self.max_seq_len)
        return question, answer, q_len, a_len

    def __len__(self):
        return self.data_len
