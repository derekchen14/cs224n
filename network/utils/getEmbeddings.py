#!/usr/bin/env python
# -*- coding: utf-8 -*-
import spacy
import csv
import os
import sys
import numpy as np
import pickle

class Config(object):
    data_path = './clean'
    minibatch_size = 20
    reduced_size = 200
    reduced_train = 120
    reduced_dev = 60
    reduced_test = 20
    test_size = 0.8
    dev_size = 0.15
    test_size = 1.0 - test_size - dev_size


def generateEmbeddingMatrix(nlp, reduced):
    config = Config()
    encodedQuestions =[]
    encodedAnswers = []
    embedding_matrix = []
    encoded = {}
    decoder = {}

    def loadQAPairs():
        print "Reading csv files."
        data = []
        for filename in os.listdir(config.data_path):
            if filename.endswith(".csv"):
                toOpen = os.path.join(config.data_path,filename)
                with open(toOpen, 'rb') as f:
                    data.extend([row for row in csv.reader(f.read().splitlines())][1:])
        print "Reading csv files complete."
        return data

    data = loadQAPairs()

    print "Generating embedding matrix."
    for i, pair in enumerate(data):
        if i % 100 == 0:
            print pair
        encodedPair = []
        if reduced:
            if i >=config.reduced_size:
                break
        assert len(pair) == 3, "Pair wrong size. %s" % pair

        for phrase in pair[:2]:
            if len(phrase) == 0:
                encodedPair = []
                print "This phrase had a Q/A of len 0. Ignoring it.", phrase
                break
            assert len(phrase) > 0, "Empty QorA."
            encodedPhrase = []
            doc = nlp(unicode(phrase, errors='ignore'))
            for word in doc:
                if str(word) not in encoded:
                    vIndex = len(encoded.keys())
                    encoded[str(word)] = vIndex
                    decoder[vIndex] = str(word)
                    embedding_matrix.append(word.vector)
                encodedPhrase.append(encoded[str(word)])
            encodedPair.append(encodedPhrase)

        if encodedPair:
            encodedQuestions.append(encodedPair[1])
            encodedAnswers.append(encodedPair[0])
        assert len(encodedQuestions) == len(encodedAnswers), 'Num answers and questions not equal. %s, %s' % (len(encodedQuestions),len(encodedAnswers))
    print "Generating embedding matrix complete."
    return encodedAnswers, encodedQuestions, embedding_matrix, decoder


def load_and_preprocess_data(reduced=True):
    config = Config()
    nlp = spacy.load('en')
    encodedAnswers, encodedQuestions, embedding_matrix, decoder = generateEmbeddingMatrix(nlp, reduced)

    # Split up the data.
    if reduced:
        a = config.reduced_train
        b = config.reduced_train+config.reduced_dev
    else:
        a = float(len(encodedAnswers))* config.test_size
        b = a + float(len(encodedAnswers))* config.dev_size
    train_set = [encodedAnswers[:a], encodedQuestions[:a]]
    dev_set = [encodedAnswers[a:b], encodedQuestions[a:b]]
    test_set = [encodedAnswers[b:], encodedQuestions[b:]]

    return np.array(train_set), np.array(dev_set), np.array(test_set), np.array(embedding_matrix), decoder

#### Minibatches ####
def get_minibatches(data, minibatch_size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)
def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

### Call this one with data #####
def get_minibatches(A,Q):
    minibatchesGen = get_minibatches([A,Q], config.minibatch_size, shuffle=True)
    # How to use:
    # for minibatch in minibatchesGen:
    #     print 'Answers in batch: ', minibatch[0]
    #     print 'Questions in batch: ', minibatch[1]
    return minibatchesGen


if __name__ == '__main__':

    train, dev, test, embedding_matrix, decoder = load_and_preprocess_data(False)
    A = train[0]
    Q = train[1]

    print 'A[:5] ', A[:5]
    print 'Q[:5] ',Q[:5]
    print embedding_matrix.shape

    Q_lens = [len(q) for q in Q]
    A_lens = [len(a) for a in A]

    import pylab as P
    P.figure()
    n, bins, patches = P.hist([Q_lens,A_lens], 20, histtype='bar')
    P.show()

    for i in range(len(Q)):
        if len(Q[i]) < 5 and len(A[i])<5:
            print 'Q:', ' | '.join([decoder[w] for w in Q[i]])
            print 'A:', ' | '.join([decoder[w] for w in A[i]])








