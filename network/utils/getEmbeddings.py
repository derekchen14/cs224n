#!/usr/bin/env python
# -*- coding: utf-8 -*-
import spacy
import csv
import os
import sys
import numpy as np
import pickle

# length includes +two for padding. Useful to decrease this number when debugging.
max_enc_dec_length = 14

def chopEnd(listToChop, lengthToChop):
    if len(listToChop) > lengthToChop-2:
        return listToChop[:(lengthToChop-2)]
    return listToChop

def chopStart(listToChop, lengthToChop):
    if len(listToChop) > lengthToChop-2:
        return listToChop[(-lengthToChop+2):]
    return listToChop

def myCustomRead(f):
    listOfStrings = f.readlines()
    listOfStrings = [l.strip().split() for l in listOfStrings]
    listOfInts = [[np.int32(l) for l in someString] for someString in listOfStrings]
    return listOfInts

def loader():

    # maximum encoder and decoder lengths shall include start/end-of-sentence padding, 2 paddings total.
    all_data = {}
    with open('autogeneratedFiles/train.ids.answers','r') as f:
        training_data_answers = myCustomRead(f)
    with open('autogeneratedFiles/train.ids.questions','r') as f:
        training_data_questions = myCustomRead(f)
    with open('autogeneratedFiles/val.ids.answers','r') as f:
        val_data_answers = myCustomRead(f)
    with open('autogeneratedFiles/val.ids.questions','r') as f:
        val_data_questions = myCustomRead(f)


    with open('autogeneratedFiles/vocab.dat') as f:
        vocabs = f.readlines()

    vocabs = [v.strip() for v in vocabs]
    all_data['vocabs_list'] = vocabs

    if True:
        training_data_questions = [chopStart(q, max_enc_dec_length) for q in training_data_questions]
        val_data_questions = [chopStart(q, max_enc_dec_length) for q in val_data_questions]
        training_data_answers = [chopEnd(q, max_enc_dec_length) for q in training_data_answers]
        val_data_answers = [chopEnd(q, max_enc_dec_length) for q in val_data_answers]

    all_data['training_data'] = [training_data_questions, training_data_answers]
    all_data['validation_data'] = [val_data_questions, val_data_answers]

    allEnc = training_data_questions + val_data_questions
    allDec = training_data_answers + val_data_answers
    enc_lengths = [len(l) for l in allEnc]
    dec_lengths = [len(l) for l in allDec]
    all_data['max_enc_len'] = max(enc_lengths) + 2
    all_data['max_dec_len'] = max(dec_lengths) + 2

    gloveNpz = np.load('autogeneratedFiles/glove.trimmed.50.npz','rb')
    all_data['embedding_matrix'] = gloveNpz['glove']

    all_data['vocab_size'] = len(all_data['embedding_matrix'])

    if False:
        import pylab as P
        P.figure()
        n, bins, patches = P.hist([enc_lengths,dec_lengths], 35, histtype='bar')
        P.show()
        print all_data['max_enc_len']
        print all_data['max_dec_len']

    return all_data


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
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
def get_batches(data, size, shuffle=True, toy=False):
    if toy:
        batchGenerator = splitToyData(data, size, shuffle)
        statistics = {"vocab_size": 27}
    else:
        batchGenerator = get_minibatches(data, size, shuffle)
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

def embedding_to_text(test_samples, final_output, lookup):
  for idx, sample_word in enumerate(test_samples):
    result = ['Prediction ', str(idx+1), ': ']

    for letter in sample_word:
      try:
        position = letter.index(1)
        result.append(lookup[position])
      except ValueError:
        result.append(' ')

    predicted_word = final_output[idx]
    for letter in predicted_word:
      big = max(letter)
      if type(letter) == list:
        position = letter.index(big)
      else:
        position = letter.tolist().index(big)

      if big > 0.5:
        result.append(lookup[position])
      elif big > 0.4:
        result.append("("+lookup[position]+")")
      else:
        result.append('-')
    print ''.join(result)

if __name__ == '__main__':
    loader()
    pass








