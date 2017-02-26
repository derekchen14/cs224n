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


def loader(reduced=True):
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
def splitData(data, mbatch_size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    print "data_size:", data_size
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for mbatch_start in np.arange(0, data_size, mbatch_size):
        mbatch_indices = indices[mbatch_start:mbatch_start + mbatch_size]
        yield [minibatch(d, mbatch_indices) for d in data] if list_data \
            else minibatch(data, mbatch_indices)
def minibatch(data, minibatch_idx):
    if type(data) is np.ndarray:
        return data[minibatch_idx]
    else:
        return [data[i] for i in minibatch_idx]

### Call this one with data #####
def get_batches(data, size, shuffle=True, toy=False):
    if toy:
        batchGenerator = splitToyData(data, size, shuffle)
        statistics = {"vocab_size": 27}
    else:
        batchGenerator = splitData(data, size, shuffle)
        statistics = 10
    # How to use:
    # for minibatch in minibatchesGen:
    #     print 'Answers in batch: ', minibatch[0]
    #     print 'Questions in batch: ', minibatch[1]
    return batchGenerator

def splitToyData(data, minibatch_size, shuffle):
    data_size = len(data)/2
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        q_start = minibatch_start
        q_end = minibatch_start + minibatch_size
        queries = [data[i] for i in np.arange(q_start, q_end)]

        a_start = minibatch_start + data_size
        a_end = minibatch_start + minibatch_size + data_size
        answers = [data[i] for i in np.arange(a_start, a_end)]

        yield [queries, answers]

def embedding_to_text(test_samples, final_output):
  lookup = list(' abcdefghijklmnopqrstuvwxyz')
  for idx, sample_word in enumerate(test_samples):
    result = ['Prediction ', str(idx+1), ': ']

    for letter in sample_word:
      try:
        position = letter.index(1)
        result.append(lookup[position])
      except ValueError:
        result.append(' ')
    result.append(' ')

    predicted_word = final_output[idx]
    for letter in predicted_word:
      big = max(letter)
      position = letter.tolist().index(big)
      if big > 0.5:
        result.append(lookup[position])
      elif big > 0.4:
        result.append("("+lookup[position]+")")
      else:
        result.append('-')
    print ''.join(result)

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








