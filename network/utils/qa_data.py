from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
hyperlink = b"https:<link>"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, hyperlink]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
hyperlink = 4

import sys
reload(sys)    
sys.setdefaultencoding('utf8')

import spacy
print('loading en for spacy')
nlp = spacy.load('en')
print('complete loading en for spacy')


def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("autogeneratedFiles")
    glove_dir = os.path.join("network", "utils", "dwr")
    source_dir = os.path.join("finalData")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=50, type=int)
    # parser.add_argument("--max_seq_len", default=100, type=int)
    return parser.parse_args()

def spacy_tokenizer(paragraph):
    # Uses spacy to parse multi-sentence paragraphs into a list of token words.
    words = []
    intermediate_words = []

    def processWord(word):
        word = str(word).lower().strip()
        if 'http' in word:
            return [hyperlink]
        # Don't clean these!
        if word in [',', '.', '?', '!', ':', '"', '\'', '/', '[', ']', '+', '-', '--', '&', '@']:
            return [word]
        m = re.findall('([\w\.\,\:\%\$\']+)', word)
        m = [w.strip('.') for w in m]
        m = [w for w in m if w]
        return m
        
    doc1 = nlp(unicode(paragraph, errors='ignore'))
    for sent in doc1.sents:
        # 2-d list.
        sentence = [processWord(word) for word in sent]
        # flatten.
        sentence = [word for words in sentence for word in words]
        sentence = [str(word) for word in sentence if word]
        words.extend(sentence)
        intermediate_words.append(sentence)

    # Make sure that none are larger than our largest bucket.
    while len(words) >= 100:
        del intermediate_words[-1]
        words = [w for s in intermediate_words for w in s]

    return words


def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def process_glove(args, vocab_list, save_path, size=4e5):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        # DEREK: These are initialized to zero vectors. 
        # If not found in glove, they remain zero vectors, and are not overwritten.
        # TODO: investigate other options. Try training them as well.
        glove = np.zeros((len(vocab_list), args.glove_dim))
        vocabNotFound = []
        not_found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                elif word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                elif word.lower() in vocab_list:
                    idx = vocab_list.index(word.lower())
                    glove[idx, :] = vector
                elif word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                else:
                    not_found += 1

        for i, vocabWord in enumerate(vocab_list):
            if sum(glove[i,:]) == 0:
                vocabNotFound.append(vocabWord)
        print(vocabNotFound)
        found = size - not_found
        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def create_vocabulary(vocabulary_path, data_paths):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = spacy_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


def sentence_to_token_ids(sentence, vocabulary):
    words = spacy_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


if __name__ == '__main__':

    # TODO: investigate having separate encoder/decoder vocabularies. 

    args = setup_args()
    vocab_path = pjoin(args.vocab_dir, "vocab.dat")

    if not os.path.exists('autogeneratedFiles'):
        os.makedirs('autogeneratedFiles')
    if not os.path.exists('finalData'):
        os.makedirs('finalData')    
        print("Derek, copy the contents from drive in finalData directory I just made for you.")
        sys.exit()

    trainAnswersSrc = pjoin(args.source_dir, "trainAnswers.txt")
    trainQuestionsSrc = pjoin(args.source_dir, "trainQuestions.txt")
    validationAnswersSrc = pjoin(args.source_dir, "validationAnswers.txt")
    validationQuestionsSrc = pjoin(args.source_dir, "validationQuestions.txt")

    create_vocabulary(vocab_path,
                      [trainAnswersSrc, trainQuestionsSrc, 
                      validationAnswersSrc, validationQuestionsSrc])
    vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below

    process_glove(args, rev_vocab, args.vocab_dir + "/glove.trimmed.{}".format(args.glove_dim))

    # ======== Creating Dataset =========
    # We created our data files seperately
    # If your model loads data differently (like in bulk)
    # You should change the below code

    # Construct destination file names.
    train_path = pjoin(args.vocab_dir, "train")
    valid_path = pjoin(args.vocab_dir, "val")
    test_path = pjoin(args.vocab_dir, "test")
    x_train_dis_path = train_path + ".ids.questions"
    y_train_ids_path = train_path + ".ids.answers"
    x_dis_path = valid_path + ".ids.questions"
    y_ids_path = valid_path + ".ids.answers"

    data_to_token_ids(trainQuestionsSrc, x_train_dis_path, vocab_path)
    data_to_token_ids(trainAnswersSrc, y_train_ids_path, vocab_path)
    data_to_token_ids(validationQuestionsSrc, x_dis_path, vocab_path)
    data_to_token_ids(validationAnswersSrc, y_ids_path, vocab_path)

    print('Done.')

