#!/usr/bin/env python
# -*- coding: utf-8 -*-
import spacy
import csv
import os
import sys
import numpy as np
import pickle
import json
import csv

reload(sys)
sys.setdefaultencoding("utf-8")


def loadDirty():
    data = []
    for filename in os.listdir('dirty/twitter'):
        if filename.endswith(".json"):
            toOpen = os.path.join('dirty/twitter',filename)
            with open(toOpen, 'rb') as f:
                readText = f.read()
            readJson = json.loads(readText)
            data.extend([r['text'] for r in readJson])
    return data

data = loadDirty()
nlp = spacy.load('en')

answers = []
questions = []
for tweet in data:
    sTweet = nlp(unicode(tweet))
    sentences = [s for s in sTweet.sents]
    if len(sentences)<2:
        continue
    elif len(sentences) == 2:
        question = sentences[0]
        answer = sentences[1]
    else:
        allLengths = [len(s) for s in sentences]
        rightSide = sum(allLengths)
        leftSide = 0
        diffs = []
        for l in allLengths:
            leftSide = leftSide + l
            rightSide = rightSide - l
            diffs.append(abs(leftSide - rightSide))
        splitIndex = diffs.index(min(diffs))
        question = ' '.join([str(s) for s in sentences[:splitIndex+1]])
        answer = ' '.join([str(s) for s in sentences[splitIndex+1:]])
    # print 'Q:',question
    # print 'A:',answer
    answers.append(answer)
    questions.append(question)


print len(answers)
print len(questions)

with open('clean/twitter.csv', 'wb+') as f:
    myWriter = csv.writer(f)
    for i in range(len(answers)):
        myWriter.writerow([answers[i], questions[i], 'twitter'])





