import json
import shared_util as su
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Choose folder where files are to be stored",type=str)
parser.add_argument("vocab_size", help="Choose size of dictionary",type=int)
parser.add_argument("process_batch_size", help="larger batch size speeds up the process but needs larger memory",type=int)

args = parser.parse_args()

if __name__ == "__main__":
    data_path = args.data_path
    train_file_in = data_path + "train.txt"
    valid_file_in = data_path + "valid.txt"
    train_file_out = data_path + "train.h5"
    valid_file_out = data_path + "valid.h5"
    test_file_in = data_path + "test.txt"
    test_file_out = data_path + "test.h5"

    dict_path = data_path + "vocab.json"
    print("Creating dictionary...")
    vocab = su.create_dict(train_file_in, args.vocab_size,args.process_batch_size)
    dict_file = open(dict_path, "w")
    dict_file.write(json.dumps(vocab))

    print("Processing Training data...")
    su.binarize(train_file_in, train_file_out, vocab ,args.process_batch_size)

    print("Processing val data...")
    su.binarize(valid_file_in, valid_file_out, vocab ,args.process_batch_size)

    print("Processing test data...")
    su.binarize(test_file_in, test_file_out, vocab ,args.process_batch_size)