import spacy
import csv
import os
import sys
import numpy as np
import pickle

reload(sys)
sys.setdefaultencoding("utf-8")

def loadQAPairs():
    data = []
    for filename in os.listdir('clean'):
        if filename.endswith(".csv"): 
            toOpen = os.path.join('clean',filename)
            with open(toOpen, 'rb') as f:
                data.extend([row for row in csv.reader(f.read().splitlines())][1:])
    return data

def encodeToEmbeddings(data, nlp):
    encodedData = []
    decoder = {}
    for i, pair in enumerate(data):
        if i % 100 == 0:
            print pair
        assert len(pair) == 3, "Pair wrong size. %s" % pair
        encodedPair = []
        for phrase in pair[0:1]:
            assert len(phrase) > 0, "Empty QorA."
            encodedPhrase = []
            doc = nlp(unicode(phrase, errors='ignore'))
            for word in doc:
                encodedPhrase.append(word.vector)
                if word not in decoder:
                    decoder[str(word)] = word.vector
            encodedPair.append(encodedPhrase)
        # may as well keep the decoded pair attached as well.
        encodedPair.append(pair)
        encodedData.append(encodedPair)
    return encodedData, decoder, None

def encodeToOneHot(data, nlp):
    encodedData = []
    decoder = {}
    encoder = {}
    for i, pair in enumerate(data):
        if i % 100 == 0:
            print pair
        assert len(pair) == 3, "Pair wrong size. %s" % pair
        encodedPair = []
        for phrase in pair[0:1]:
            assert len(phrase) > 0, "Empty QorA."
            encodedPhrase = []
            doc = nlp(unicode(phrase, errors='ignore'))
            for word in doc:
                if word not in decoder:
                    vIndex = len(decoder.keys()) + 1
                    encoder[str(word)] = vIndex
                    decoder[vIndex] = str(word)
                encodedPhrase.append(encoder[str(word)])
            encodedPair.append(encodedPhrase)
        # may as well keep the decoded pair attached as well.
        encodedPair.append(pair)
        encodedData.append(encodedPair)
    return encodedData, decoder, encoder


data = loadQAPairs()
nlp = spacy.load('en')
if True:
    writeTo = os.path.join('clean','encodedDataEmbeddings.p')
    encodedData, decoder, __ = encodeToEmbeddings(data, nlp)
    pickle.dump(encodedData, open(writeTo,"wb+"))
    pickle.dump(decoder, open(writeTo,"wb"))
    # loadedDecoder = pickle.load(open(writeTo, "rb"))
    # loadedData = pickle.load(open(writeTo, "rb"))
else:
    writeTo = os.path.join('clean','encodedDataOneHot.p')
    encodedData, decoder, encoder = encodeToOneHot(data, nlp)
    pickle.dump(encodedData, open(writeTo,"wb+"))
    pickle.dump(decoder, open(writeTo,"wb"))
    pickle.dump(encoder, open(writeTo,"wb"))
    # loadedEncoder = pickle.load(open(writeTo, "rb"))
    # loadedDecoder = pickle.load(open(writeTo, "rb"))
    # loadedData = pickle.load(open(writeTo, "rb"))








