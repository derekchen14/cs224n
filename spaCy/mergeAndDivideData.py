

import csv
import os
import sys
import numpy as np
import pickle


def loadQAPairs():
    print "Reading csv files."
    data = []
    for filename in os.listdir('./clean'):
        if filename.endswith(".csv"):
            toOpen = os.path.join('./clean',filename)
            with open(toOpen, 'rb') as f:
                data.extend([row for row in csv.reader(f.read().splitlines())])
    print "Reading csv files complete."
    return data

data = loadQAPairs()
allQs = [d[1] for d in data]
allAs = [d[0] for d in data]

assert len(allQs) == len(allAs), 'lengths must be equal.'

lengthAll = len(allQs)
lengthTest = int(float(lengthAll) * 0.1)
lengthValidation = int(float(lengthAll) * 0.2)
lengthTrain = lengthAll - lengthTest - lengthValidation

# OVERWRITE:
lengthTrain = 12800
lengthValidation = 2560
lengthTest = lengthAll - lengthTrain - lengthValidation

trainIndexes = np.random.choice(lengthAll, lengthTrain,replace=False)
trainQuestions= [allQs[i] for i in trainIndexes]
trainAnswers = [allAs[i] for i in trainIndexes]
remainingQs = [allQs[i] for i in range(len(allQs)) if i not in trainIndexes]
remainingAs = [allAs[i] for i in range(len(allAs)) if i not in trainIndexes]

validationIndexes = np.random.choice(len(remainingAs), lengthValidation,replace=False)
validationQuestions = [remainingQs[i] for i in validationIndexes]
validationAnswers = [remainingAs[i] for i in validationIndexes]
testQuestions = [remainingQs[i] for i in range(len(remainingQs)) if i not in validationIndexes]
testAnswers = [remainingAs[i] for i in range(len(remainingAs)) if i not in validationIndexes]

assert len(testQuestions) == len(testAnswers), 'lengths must be equal.'
assert len(validationQuestions) == len(validationAnswers), 'lengths must be equal.'
assert len(trainQuestions) == len(trainAnswers), 'lengths must be equal.'

# print len(testQuestions)
# print len(validationQuestions)
# print len(trainQuestions)

with open('testQuestions.txt', 'w+') as f:
	f.write("\n".join(testQuestions))
with open('testAnswers.txt', 'w+') as f:
	f.write("\n".join(testAnswers))
with open('validationQuestions.txt', 'w+') as f:
	f.write("\n".join(validationQuestions))
with open('validationAnswers.txt', 'w+') as f:
	f.write("\n".join(validationAnswers))
with open('trainQuestions.txt', 'w+') as f:
	f.write("\n".join(trainQuestions))
with open('trainAnswers.txt', 'w+') as f:
	f.write("\n".join(trainAnswers))





