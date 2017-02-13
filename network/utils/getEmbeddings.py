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

def generateEmbeddingMatrix(nlp, reduced):
    config = Config()
    encodedQuestions =[]
    encodedAnswers = []
    embedding_matrix = []
    encoded = {}

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
            if i >=200:
                break
        assert len(pair) == 3, "Pair wrong size. %s" % pair

        for phrase in pair[:2]:
            assert len(phrase) > 0, "Empty QorA."
            encodedPhrase = []
            doc = nlp(unicode(phrase, errors='ignore'))
            for word in doc:
                if str(word) not in encoded:
                    vIndex = len(encoded.keys())
                    encoded[str(word)] = vIndex
                    embedding_matrix.append(word.vector)
                encodedPhrase.append(encoded[str(word)])
            encodedPair.append(encodedPhrase)

        if encodedPair:
            encodedQuestions.append(encodedPair[1])
            encodedAnswers.append(encodedPair[0])
        assert len(encodedQuestions) == len(encodedAnswers), 'Num answers and questions not equal. %s, %s' % (len(encodedQuestions),len(encodedAnswers))
    print "Generating embedding matrix complete."
    return encodedAnswers, encodedQuestions, embedding_matrix


def load_and_preprocess_data(reduced=True):
    nlp = spacy.load('en')
    encodedAnswers, encodedQuestions, embedding_matrix = generateEmbeddingMatrix(nlp, reduced)
    return np.array(encodedAnswers), np.array(encodedQuestions), np.array(embedding_matrix)

if __name__ == '__main__':
    A, Q, embedding_matrix = load_and_preprocess_data(True)
    print 'A[:5] ', A[:5]
    print 'Q[:5] ',Q[:5]
    print embedding_matrix.shape






