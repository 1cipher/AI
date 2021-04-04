from sklearn.feature_extraction.text import CountVectorizer
from dataset_utils import Tokenizer as lt
from nltk import word_tokenize


def setBagofWordsM(train_data,test_data):
    vectorizer = CountVectorizer(tokenizer=lt.Tokenizer() ,lowercase=False)     #prepare bag of words for multinomial classifier
    prepared_train = vectorizer.fit_transform(train_data)
    prepared_test = vectorizer.transform(test_data)
    return prepared_train,prepared_test,vectorizer


def setBagofWordsB(train_data,test_data):
    vectorizer = CountVectorizer(tokenizer=lt.Tokenizer(),lowercase=False,binary=True)     #prepare bag of words for Bernoulli classifier
    prepared_train = vectorizer.fit_transform(train_data)
    prepared_test = vectorizer.transform(test_data)
    return prepared_train,prepared_test,vectorizer