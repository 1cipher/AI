import numpy as np
import scipy as sp
from classifier import PriorCalculator as pc
import math

class BernoulliClassifier():

    def __init__(self):
        self.trainvocabulary = None
        self.condProb = None
        self.condProb2 = None
        self.Prior = None


    def train(self,vector,target,vocabulary):
        keys = list(vocabulary.keys())
        wordInPositiveDoc = [0 for j in range(len(keys))]
        wordInNegativeDoc = [0 for j in range(len(keys))]
        posP,negP,Npos,Nneg =pc.calculatePrior(target)
        N = [Nneg , Npos]
        self.Prior = [negP , posP]
        i = 0
        for i in range(vector.shape[0]):
            print(i)
            selectedDocument = vector[i].nonzero()
            for word in selectedDocument[1]:
                if(target[i] == 0):
                    wordInNegativeDoc[word] = wordInNegativeDoc[word] + 1
                else:
                    wordInPositiveDoc[word] = wordInPositiveDoc[word] + 1


        WordNegativeProbabilities = [0 for j in range(len(keys))]
        WordPositiveProbabilities = [0 for j in range(len(keys))]
        WordNegativeProbabilities2 = [0 for j in range(len(keys))]
        WordPositiveProbabilities2 = [0 for j in range(len(keys))]

        for j in range(len(WordNegativeProbabilities)):
            WordNegativeProbabilities[j] = (wordInNegativeDoc[j] + 1)/(N[0] + 2)     #calculates likelihood for positive and negative class
            WordNegativeProbabilities2[j] = -(math.log10(1 - ((wordInNegativeDoc[j] + 1) / (N[0] + 2))))

        for j in range(len(WordPositiveProbabilities)):
            WordPositiveProbabilities[j] = (wordInPositiveDoc[j] + 1)/(N[1] + 2)
            WordPositiveProbabilities2[j] = - (math.log10( 1 - ((wordInPositiveDoc[j] + 1) / (N[1] + 2))))

        print('fine train')


        self.trainvocabulary = vocabulary
        self.condProb = [WordNegativeProbabilities,WordPositiveProbabilities]
        self.condProb2 = [WordNegativeProbabilities2, WordPositiveProbabilities2]


    def predict(self,dataset):   #dataset represent test data (ae: dataset[1][2]= 0 means that the second word of V is not present in the second document of dataset)
        predictions = []
        keys = list(self.trainvocabulary.keys())
        voc = self.trainvocabulary
        condprob = self.condProb
        condprob2 = self.condProb2
        vocabularySet = set(range(0,len(self.trainvocabulary)))

        for i in range(dataset.shape[0]):
            score = [0 for i in range(len(self.Prior))]
            #apply posterior for each document and establish if it is more likely of class 0 or 1
            print('predicting ',i,' out of ',dataset.shape[0])
            selectedDocument = dataset[i].nonzero()
            inset = set(list(selectedDocument[1]))
            outset = vocabularySet-inset

            for word in selectedDocument[1]:
                p0 = condprob2[0][word]
                p1 = condprob2[1][word]
                score[0] = score[0] + (p0)
                score[1] = score[1] + (p1)

            p0 = np.sum(condprob2[0])
            p1 = np.sum(condprob2[1])

            score[0] = p0 - score[0]
            score[1] = p1 - score[1]

            for word in selectedDocument[1]:
                p0 = -(math.log10(condprob[0][word]))
                p1 = -(math.log10(condprob[1][word]))
                score[0] = score[0] + p0
                score[1] = score[1] + p1

            for i in range(len(self.Prior)):
                score[i] = score[i] - (math.log10(self.Prior[i]))

            if(score[1]>score[0]):
                predictions.append(0)
            else:
                predictions.append(1)

        return predictions

