from sklearn.datasets import load_files

train_set = load_files(r'C:\Users\Utente\Desktop\AI\test1')
train_data,train_target = train_set.data, train_set.target
print('train dataset caricati')
test_set = load_files(r'C:\Users\Utente\Desktop\AI\test2')
test_data,test_target = test_set.data,test_set.target
print('test dataset caricati')


import scipy as sp
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import math
from sklearn import metrics
import re
from nltk import word_tokenize
from numpy import matrix
from progress.bar import Bar

def Tokenization(dataset):
    for j in range(0,len(dataset)):
        content = nltk.word_tokenize(dataset[j])             #tokenize on document(not used)
        dataset[j] = content
    return dataset


def LowerCase(dataset):
    for j in range(0,len(dataset)):
        for i in range(0,len(dataset[j])):                   #choose the lower case for words(not used)
            dataset[j][i].lower()

def removePunctuationAndStopwords(dataset):
    for j in range(0, len(dataset)):
        content = dataset[j]
        dataset[j] = [word for word in dataset[j] if (word.isalpha() and word not in stopwords.words('english'))]    #(not used)


def Stemming(dataset):
    porter = PorterStemmer()
    for j in range(0, len(dataset)):
        dataset[j] = [porter.stem(word) for word in dataset[j]]   #stem process(not used)


class LemmaTokenizer(object):
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))             #Core function for tokenization
    def __call__(self, doc):
        stemmer = PorterStemmer()
        clean = [self.regex.sub('',w) for w in word_tokenize(doc)]
        return [stemmer.stem(t) for t in clean if( len(t)>2 and t not in stopwords.words('english'))]




def report(target, pred, target_names):
    cm = metrics.confusion_matrix(target, pred)
    print('Contingency matrix: (rows: examples of a given true class)')
    print(cm)
    print('Classification report')
    print(metrics.classification_report(target, pred,
                                        target_names=target_names))



def calculatePrior(label):
    pos = 0
    neg = 0
    for i in range(len(label)):
        if label[i] == 1:
            pos += 1
        else:
            neg += 1
    Ppos = pos/len(label)
    Pneg = neg/len(label)
    return Ppos,Pneg,pos,neg


def wordInClassesCounterB(word,vector,target,check):
    N = 0
    for i in range(vector.shape[0]):
        selectedDocument = vector[i].todense()
        selectedDocument = np.squeeze(np.asarray(selectedDocument))       #count how many documents of class check contain the word
        if target[i] == check and selectedDocument[word] != 0:
            N+=1
    return N


def wordInClassesCounterM(word,vector,target,check):
    N = 0
    for i in range(vector.shape[0]):
        selectedDocument = vector[i].todense()                           #count the number of word occurrence in check class documents
        selectedDocument = np.squeeze(np.asarray(selectedDocument))
        if target[i] == check and selectedDocument[word] != 0:
            N += selectedDocument[word]
    return N


def checkPresence(word,list):
     if list[word] != 0:
         return 1
     else:
         return 0

class BernoulliClassifier():

    def __init__(self):
        self.trainvocabulary = None
        self.condProb = None
        self.Prior = None


    def train(self,vector,target,vocabulary):
        keys = list(vocabulary.keys())
        WordinPositiveDoc = [0 for j in range(len(keys))]
        WordinNegativeDoc = [0 for j in range(len(keys))]
        posP,negP,Npos,Nneg = calculatePrior(target)
        N = [Nneg , Npos]
        self.Prior = [negP , posP]
        i = 0
        for key in keys:
            word = vocabulary[key]
            print('fatto: ',i,'/',len(vocabulary))
            i = i + 1
            WordinNegativeDoc[word] = wordInClassesCounterB(word,vector,target,0)     #target represent train_target,classes of doument
            WordinPositiveDoc[word] = wordInClassesCounterB(word, vector, target, 1)     #store the number of document that contains word of class 0/1

        WordNegativeProbabilities = [0 for j in range(len(keys))]
        WordPositiveProbabilities = [0 for j in range(len(keys))]
        for j in range(len(WordNegativeProbabilities)):
            WordNegativeProbabilities[j] = (WordinNegativeDoc[j] + 1)/(N[0] + 2)     #calculates likelihood for positive and negative class


        for j in range(len(WordPositiveProbabilities)):
            WordPositiveProbabilities[j] = (WordinPositiveDoc[j] + 1)/(N[1] + 2)


        self.trainvocabulary = vocabulary
        self.condProb = [WordNegativeProbabilities,WordPositiveProbabilities]


    def predict(self,dataset):   #dataset represent test data (ae: dataset[1][2]= 0 means that the second word of V is not present in the second document of dataset)
        predictions = []
        keys = list(self.trainvocabulary.keys())
        voc = self.trainvocabulary
        condprob = self.condProb

        for i in range(dataset.shape[0]):
            score = [self.Prior[i] for i in range(len(self.Prior))]    #apply posterior for each document and establish if it is more likely of class 0 or 1
            for k in range(2):
                for j in range(len(keys)):
                    selectedDocument = dataset[i].todense()
                    selectedDocument = np.squeeze(np.asarray(selectedDocument))
                    b = checkPresence(j,selectedDocument)
                    p = condprob[k][j]
                    score[k] = score[k]*((b*p+(1-b)*(1-p)))
            if(score[1]>score[0]):
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions



class MultinomialClassifier:
    def __init__(self):
        self.vocabulary = None
        self.condProb = None
        self.Prior = None

    def train(self,vector,target,vocabulary):
        keys = list(vocabulary.keys())
        WordinPositiveDoc = [0 for j in range(len(keys))]
        WordinNegativeDoc = [0 for j in range(len(keys))]
        posP, negP, Npos, Nneg = calculatePrior(target)
        N = [Nneg, Npos]
        self.Prior = [negP , posP]

        i = 0
        for key in keys:
            word = vocabulary[key]
            WordinNegativeDoc[word] = wordInClassesCounterM(word, vector, target,0)  # target represent train_target,classes of doument
            WordinPositiveDoc[word] = wordInClassesCounterM(word, vector, target, 1)   #count for each word the number of occurences in class k's docs (nk(wt))
            print('fatto: ',i,'/',len(vocabulary))
            i = i + 1

        WordNegativeProbabilities = [0 for j in range(len(keys))]
        WordPositiveProbabilities = [0 for j in range(len(keys))]

        totalWordofClassPos = 0
        totalWordofClassNeg = 0                                           #count of total occurences of words in document of class 1/0 (sum on s(nk(ws))
        for i in range(len(vocabulary)):
            totalWordofClassNeg = totalWordofClassNeg + WordinNegativeDoc[i]
            totalWordofClassPos = totalWordofClassPos + WordinPositiveDoc[i]

        totalWordofClassK = [totalWordofClassNeg,totalWordofClassPos]

        for j in range(len(WordNegativeProbabilities)):                   #calculates likelihoods
            WordNegativeProbabilities[j] = (WordinNegativeDoc[j] + 1) / (totalWordofClassK[0] + len(keys))

        for j in range(len(WordPositiveProbabilities)):
            WordPositiveProbabilities[j] = (WordinPositiveDoc[j] + 1) / (totalWordofClassK[1] + len(keys))

        print('fine apprendimento')

        self.vocabulary = vocabulary
        self.condProb = [WordNegativeProbabilities, WordPositiveProbabilities]


    def predict(self,dataset):

        predictions = []
        condProb = self.condProb

        for i in range(dataset.shape[0]):                             #calculates predictions for each document
            score = [self.Prior[i] for i in range(len(self.Prior))]
            for k in range(2):
                for j in range(len(self.vocabulary)):
                    selectedDocument = dataset[i].todense()
                    selectedDocument = np.squeeze(np.asarray(selectedDocument))
                    xt = selectedDocument[j]
                    p = condProb[k][j]
                    score[k] = score[k]*(pow(p,xt))

            if(score[0]>score[1]):
                predictions.append(0)
            else:
                predictions.append(1)

        return predictions


def prepareDataset(data):
    for j in range(len(data)):
        content = data[j].decode('utf-8')  #need to convert from a utf-8 string to unicode string in order to do the process (not used)
        data[j] = content

    data = Tokenization(data)
    removePunctuationAndStopwords(data)
    Stemming(data)
    return data

def setBagofWordsM(train_data,test_data):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer() ,lowercase=False)     #prepare bag of words for multinomial classifier
    prepared_train = vectorizer.fit_transform(train_data)
    print('lunghezza del vocabolario:',len(vectorizer.vocabulary_))
    print(vectorizer.vocabulary_)
    prepared_test = vectorizer.transform(test_data)
    return prepared_train,prepared_test,vectorizer


def setBagofWordsB(train_data,test_data):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),lowercase=False,binary=True)     #prepare bag of words for Bernoulli classifier
    prepared_train = vectorizer.fit_transform(train_data)
    print('lunghezza del vocabolario:',len(vectorizer.vocabulary_))
    print(vectorizer.vocabulary_)
    a = prepared_train[1].todense()
    r = np.squeeze(np.asarray(a))
    print(len(r))
    print(type(r))
    print(r.size)
    print(type(prepared_train))
    prepared_test = vectorizer.transform(test_data)
    return prepared_train,prepared_test,vectorizer

#train = prepareDataset(train_data)
#print('train dataset pronto')        #how was after introducing LemmaTokenizer
#test = prepareDataset(test_data)
train,test,vectorizer = setBagofWordsM(train_data,test_data)

print('test dataset pronto')

clf = MultinomialClassifier()
clf.train(train,train_target,vectorizer.vocabulary_)
pred = clf.predict(test)
print(pred)
print(test_target)
print((len(pred)))
print(len(test_target))
report(test_target,pred,['neg','pos'])

print('FINE MULTINOMIAL')

train1,test1,vectorizer1 = setBagofWordsB(train_data,test_data)

clf1 = BernoulliClassifier()
clf1.train(train1,train_target,vectorizer1.vocabulary_)
pred1 = clf1.predict(test1)
print(pred1)
print(test_target)
print((len(pred1)))
print(len(test_target))
report(test_target,pred1,['neg','pos'])


