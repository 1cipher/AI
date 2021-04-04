import scipy as sp
import numpy as np
from dataset_utils import load
from classifier import BagOfWordsGenerator as b
from classifier import MultinomialNaiveBayes as m
from dataset_utils import report as r



train = sp.sparse.load_npz("trainMultinomial.npz")
test = sp.sparse.load_npz("testMultinomial.npz")

trainTarget = np.load("trainTarget.npz")['arr_0']
testTarget = np.load("testTarget.npz")['arr_0']
vectorizer = np.load("vectorizerVocabularyM.npz",allow_pickle=True)['arr_0.npy']
dict = vectorizer.tolist()

clf = m.MultinomialClassifier()
clf.train(train,trainTarget,dict)
pred = clf.predict(test)

r.report(testTarget,pred,['neg','pos'])


