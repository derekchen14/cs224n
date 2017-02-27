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
import re

reload(sys)
sys.setdefaultencoding("utf-8")

print 'spacy..'
nlp = spacy.load('en')
print 'done spacy'


discardWords = set(['interview', 'book', 'signing','pageant','@todayshow','apprentice', '@missuniverse','article','unveiling','album','hosted', 'narrated'])
keepWords = set(['@barackobama', 'obama', '.@barackobama', 'europe', 'jobs', 'job', 'oil', 'pipeline', 'tax', 'taxes', 'taxing', 'un', 'isreal', 'washington',
    'afghanistan', 'veterans','car', 'americans', 'elect', 'presidential', 'nomination', 'negotiation','deal','business', 'owners',
    'ventures', 'violence','country','republicans', 'debt','china', 'buying','government', 'borrowing','legal','rigged', 'claims','stock',
    'donated','campaign','rioting','gasoline', 'gas','votestand', 'votes', 'illegal','obamacare','nuclear','weapon', 'mexico', 'election',
    'speech','republican','nominations','iraq','political', 'turmoil','sanders','debate','u.s.a', 'supporters','president','bankrupt','tirade',
    'dishonest', 'scum','financial', 'crisis','entrepreneur', 'war','price','russia','hillary', 'clinton','terror', 'education', 'roads','nafta','manufacturing',
    'killing','kill','millions','million','billions','billion','environment','newspaper','global', 'power','war','communism','senator','polls','profit','presbyterian',
    'putin', 'enemies','dishonest', 'media','missile', 'defense', 'systems','military','resign','settlement','market','country','countries','spying','drugs',
    'u.s.','rape','border','wall','corrupt', 'clintons','russian', 'hacking','bankruptcy','construction','pervert','gifted', 'politician','candidate','crookedhillary','hacked'
    'disaster','dangerous','battlefield','deductibles','poll','loans','canada','congress','conference','violation','fraud','ads','deceit','poll','democrats',
     'secret', 'intelligence','investigate','murder','shootings','sector','unreported', 'drug','salary','benghazi','balance', 'budget', 'amendment',
     'italy', 'recognized', 'innocence','consumer', 'spending','us', 'cyber', 'espionage', 'charges', 'immigration','eight', 'syrians', 'caught', 'border', 'isis'])

def loadDirty():
    data = []
    for filename in os.listdir('dirty/twitter'):
        if filename.endswith(".json") and filename not in ['2009.json', '2010.json']:
            toOpen = os.path.join('dirty/twitter',filename)
            with open(toOpen, 'rb') as f:
                readText = f.read()
            readJson = json.loads(readText)
            data.extend([r['text'] for r in readJson])
    return data

def qualifies(d):
    global keepWords
    global discardWords

    if d[0] == '"':
        return None, False

    d = re.sub(r'^https?:\/\/.*[\r\n]*', '', d, flags=re.MULTILINE)

    tweet = nlp(unicode(d))
    if len(tweet)<3:
        return None, False

    tweetedPhrase = []
    for sent in tweet.sents:
        for word in sent:
            appendMe = True
            if str(word) == 'cont':
                appendMe = False
                break
        if appendMe:
            tweetedPhrase.extend([str(w).lower() for w in sent])

    if 'cont' in tweetedPhrase:
        print "Shoudn't be in here. EEEERRRROROROORORO"
        print tweetedPhrase
        return None, False

    if not tweetedPhrase:
        return None, False
    
    tweetedSet = set(tweetedPhrase)
    overlapWithKeep = tweetedSet & keepWords
    overlapWithDiscard = tweetedSet & discardWords

    if overlapWithDiscard:
        return None, False
    if overlapWithKeep:
        return tweetedPhrase, True

    print "needs classifying: ", tweetedPhrase
    return None, False


data = loadDirty()
dataToWrite = []
allAnswers = []
for idx,d in enumerate(data):
    outSentence, keep = qualifies(d)
    if keep:
        allAnswers.append(outSentence)

print "Length Out: ", len(allAnswers)

allQuestions = []
QA_Data = []
for a in allAnswers:
    QA_Data.append([a,set(a) & keepWords, set(a) - keepWords])
for answer, keeps, others in QA_Data:
    similarityCounts = []
    for potentialIndex, (potentialQ, potentialKeep, potentialOthers) in enumerate(QA_Data):
        similarityCounts.append([(len(keeps & potentialKeep), len(others & potentialOthers)),potentialIndex])
    # Sort based on tuple
    similarityCounts = sorted(similarityCounts, key=lambda s : s[0],reverse=True)

    print 'questions: ', QA_Data[similarityCounts[1][1]][0]
    print 'answer: ', answer

    allQuestions.append(QA_Data[similarityCounts[1][1]][0])


assert len(allQuestions) == len(allAnswers), 'length should equal'


with open("allTwitter.csv", "w+") as f:
    writer = csv.writer(f)
    for idx in range(len(allQuestions)):
        writer.writerow([' '.join(allAnswers[idx]),' '.join(allQuestions[idx])])




