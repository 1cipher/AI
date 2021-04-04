import scipy as sp
import numpy as np
from dataset_utils import load
from classifier import BernoulliNaiveBayes as b
from dataset_utils import report as r



train = sp.sparse.load_npz("trainBernoulli.npz")
test = sp.sparse.load_npz("testBernoulli.npz")

trainTarget = np.load("trainTarget.npz")['arr_0']
testTarget = np.load("testTarget.npz")['arr_0']
vectorizer = np.load("vectorizerVocabulary.npz",allow_pickle=True)['arr_0.npy']
dict = vectorizer.tolist()

clf = b.BernoulliClassifier()
clf.train(train,trainTarget,dict)
pred = clf.predict(test)

r.report(testTarget,pred,['neg','pos'])