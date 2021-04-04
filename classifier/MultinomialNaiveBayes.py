import numpy as np
import scipy as sp
from classifier import PriorCalculator as pc
import math

class MultinomialClassifier:
    def __init__(self):
        self.vocabulary = None
        self.condProb = None
        self.Prior = None

    def train(self,vector,target,vocabulary):
        keys = list(vocabulary.keys())
        wordInPositiveDoc = [0 for j in range(len(keys))]
        wordInNegativeDoc = [0 for j in range(len(keys))]
        posP, negP, Npos, Nneg = pc.calculatePrior(target)
        N = [Nneg, Npos]
        self.Prior = [negP , posP]

        i = 0
        for i in range(vector.shape[0]):
            print('Training ',i+1,' out of ',vector.shape[0])
            selectedDocument = vector[i].nonzero()
            bagOfWords = np.squeeze(np.asarray(vector[i].todense()))

            for word in selectedDocument[1]:
                if(target[i] == 0):
                    wordInNegativeDoc[word] = wordInNegativeDoc[word] + bagOfWords[word]
                else:
                    wordInPositiveDoc[word] = wordInPositiveDoc[word] + bagOfWords[word]


        wordNegativeProbabilities = [0 for j in range(len(keys))]
        wordPositiveProbabilities = [0 for j in range(len(keys))]

        totalWordOfClassPos = 0
        totalWordOfClassNeg = 0                                           #count of total occurences of words in document of class 1/0 (sum on s(nk(ws))
        for i in range(len(vocabulary)):
            totalWordOfClassNeg = totalWordOfClassNeg + wordInNegativeDoc[i]
            totalWordOfClassPos = totalWordOfClassPos + wordInPositiveDoc[i]

        totalWordOfClassK = [totalWordOfClassNeg,totalWordOfClassPos]

        for j in range(len(wordNegativeProbabilities)):                   #calculates likelihoods
            wordNegativeProbabilities[j] = (wordInNegativeDoc[j] + 1) / (totalWordOfClassK[0] + len(keys))

        for j in range(len(wordPositiveProbabilities)):
            wordPositiveProbabilities[j] = (wordInPositiveDoc[j] + 1) / (totalWordOfClassK[1] + len(keys))

        print('fine apprendimento')

        self.vocabulary = vocabulary
        self.condProb = [wordNegativeProbabilities, wordPositiveProbabilities]



    def predict(self,dataset):

        predictions = []
        score1 = []
        condProb = self.condProb

        for i in range(dataset.shape[0]):                             #calculates predictions for each document
            score = [-(math.log10(self.Prior[i])) for i in range(len(self.Prior))]
            selectedDocument = np.squeeze(np.asarray(dataset[i].todense()))
            bagOfWords = dataset[i].nonzero()
            print('predicting ', i, ' out of ', dataset.shape[0])

            for k in range(2):
                for word in bagOfWords[1]:
                    xt = selectedDocument[word]
                    p = condProb[k][word]
                    score[k] = score[k] - (math.log10(pow(p,xt)))

            if(score[0]>score[1]):
                predictions.append(1)
            else:
                predictions.append(0)
        print(score)
        return predictions
