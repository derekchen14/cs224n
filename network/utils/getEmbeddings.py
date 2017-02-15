#!/usr/bin/env python
# -*- coding: utf-8 -*-
import spacy
import csv
import os
import sys
import numpy as np
import pickle


def generateEmbeddingMatrix(nlp, config, reduced):
    encodedQuestions =[]
    encodedAnswers = []
    embedding_matrix = []
    encoded = {}
    decoder = {}

    def loadQAPairs(reduced):
        print "Reading csv files."
        data = []
        for filename in os.listdir(config.data_path):
            if filename.endswith(".csv"):
                toOpen = os.path.join(config.data_path,filename)
                with open(toOpen, 'rb') as f:
                    if reduced and len(data) >=config.reduced_size:
                        break
                    data.extend([row for row in csv.reader(f.read().splitlines())][1:])
        print "Reading csv files complete."
        return data

    data = loadQAPairs(reduced)

    print "Generating embedding matrix."
    for index, pair in enumerate(data):
        if reduced and (index >= config.reduced_size):
            break

        encodedPair = []
        for phrase in pair[:2]:
            assert len(phrase) > 0, "Empty QorA."
            encodedPhrase = []
            doc = nlp(unicode(phrase, errors='ignore'))
            for word in doc:
                if str(word) not in encoded:
                    vIndex = len(encoded.keys())
                    encoded[str(word)] = int(vIndex)
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


def loader(config, reduced):
    nlp = spacy.load('en')
    encodedAnswers, encodedQuestions, embedding_matrix, decoder = generateEmbeddingMatrix(nlp, config, reduced)

    # Split up the data.
    if reduced:
        a = config.reduced_train
        b = config.reduced_train+config.reduced_dev
        c = config.reduced_size
    else:
        a = int(float(len(encodedAnswers))* config.test_size)
        b = int(a + float(len(encodedAnswers))* config.dev_size)

    train_set = [encodedAnswers[:a], encodedQuestions[:a]]
    dev_set = [encodedAnswers[a:b], encodedQuestions[a:b]]
    test_set = [encodedAnswers[b:], encodedQuestions[b:]]

    matrix = np.asarray(embedding_matrix, dtype='float32')
    return train_set, dev_set, test_set, matrix, decoder

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

### Call like this, with data #####
# def get_minibatches(A,Q):
#     minibatchesGen = get_minibatches([A,Q], config.minibatch_size, shuffle=True)
#     # How to use:
#     # for minibatch in minibatchesGen:
#     #     print 'Answers in batch: ', minibatch[0]
#     #     print 'Questions in batch: ', minibatch[1]
#     return minibatchesGen


if __name__ == '__main__':

    ## BROKEN NOW , needs config. ##

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

    config = Config()
    train, dev, test, embedding_matrix, decoder = loader(config, reduced=False)
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








